import numpy as np

from lotto_nn.prime_pipeline import (
    FIRST_10_PRIMES,
    decode_draws,
    encode_draws,
    run_prime_pipeline,
)


def test_encode_decode_roundtrip():
    draws = np.array([[1, 2, 3, 4, 5, 59], [7, 13, 21, 33, 48, 58]])
    for base in FIRST_10_PRIMES:
        assert (decode_draws(encode_draws(draws, base).numpy(), base) == draws).all()


def test_run_prime_pipeline(tmp_path):
    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    for i in range(12):
        nums = [(n + i * 3) % 59 + 1 for n in (1, 9, 17, 25, 33, 41)]
        rows.append(f"2022-{i + 1:02d}-01,{','.join(map(str, sorted(nums)))},7")
    csv = tmp_path / "d.csv"
    csv.write_text("\n".join(rows) + "\n")
    out = tmp_path / "o.txt"
    res = run_prime_pipeline(str(csv), epochs=1, batch_size=4, num_samples=5, verbose=False, out=str(out))
    assert list(res) == FIRST_10_PRIMES
    for r in res.values():
        g = r["generated_np"]
        assert g.shape == (5, 6)
        assert ((g >= 1) & (g <= 59)).all()
        assert all(len(set(row)) == 6 for row in g)
    assert len(out.read_text().splitlines()) == 10
