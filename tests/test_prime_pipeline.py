import numpy as np

from lotto_nn.prime_pipeline import (
    FIRST_10_PRIMES,
    run_prime_pipeline,
)


def test_run_prime_pipeline(tmp_path):
    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    for i in range(12):
        nums = [(n + i * 3) % 59 + 1 for n in (1, 9, 17, 25, 33, 41)]
        rows.append(f"2022-{i + 1:02d}-01,{','.join(map(str, sorted(nums)))},7")
    csv = tmp_path / "d.csv"
    csv.write_text("\n".join(rows) + "\n")
    res = run_prime_pipeline(str(csv), epochs=1, batch_size=4, num_samples=5, verbose=False)
    assert list(res) == FIRST_10_PRIMES
    for r in res.values():
        g = r["generated_np"]
        assert g.shape == (5, 6)
        assert ((g >= 1) & (g <= 59)).all()
        assert all(len(set(row)) == 6 for row in g)


def test_batched_digit_sampling_gives_valid_draws():
    import numpy as np
    import torch
    from lotto_nn.model import LottoTransformer
    from lotto_nn.prime_pipeline import NUMS_PER_DRAW, digit_width, generate_batch_base

    for base in (2, 7, 29):
        torch.manual_seed(0)
        model = LottoTransformer(vocab_size=base, seq_len=NUMS_PER_DRAW * digit_width(base))
        draws = generate_batch_base(model, base, 300)
        assert draws.shape == (300, 6)
        assert all(len(set(d)) == 6 and 1 <= min(d) and max(d) <= 59 for d in draws)
