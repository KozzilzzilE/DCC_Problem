"""T9 전용 학습 목적함수. 추론 코드에서는 사용하지 않는다."""
import torch


def bernoulli_symmetric_kl(z1, z2):
    """양방향 Gradient를 갖는 Bernoulli 대칭 KL의 Batch×증상 mean."""
    with torch.autocast(device_type=z1.device.type, enabled=False):
        a, b = z1.float(), z2.float()
        return (0.5 * (a.sigmoid() - b.sigmoid()) * (a - b)).mean()


def prefix_objective(model, enc, target, loss_fn, consistency_lambda=0.0):
    """동일 Tensor 입력을 순차 호출한다. RNG 상태를 저장/복원하지 않는다."""
    device_type = target.device.type
    with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
        z1 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        if consistency_lambda > 0:
            z2 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    n_sym = target.shape[-1]
    with torch.autocast(device_type, enabled=False):
        supervised = loss_fn(z1.float()[:, :n_sym], target)
        raw = supervised.new_zeros(())
        if consistency_lambda > 0:
            supervised = (supervised + loss_fn(z2.float()[:, :n_sym], target)) / 2
            raw = bernoulli_symmetric_kl(z1[:, :n_sym], z2[:, :n_sym])
        total = supervised + consistency_lambda * raw if consistency_lambda > 0 else supervised
    return total, supervised, raw, z1


def validate_t9(args):
    if args.consistency_lambda == 0:
        return
    expected = dict(consistency_lambda=1.0, pos_weight_power=0.5, grad_ckpt=True,
                    batch_size=16, grad_accum=1, epochs=3, seed=42, samples_per_call=2,
                    include_full=False, count_classes=0, max_length=512, fmt="space",
                    truncation_side="right", lr=5e-5, llrd=0.8, max_steps=0)
    changed = [k for k,v in expected.items() if getattr(args, k, None) != v]
    if changed:
        raise ValueError(f"확정 T9 설정과 다릅니다: {changed}. Batch/학습량을 자동 변경하지 않습니다")
