from lotto_nn.pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    import textwrap

    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    base_numbers = [1, 2, 3, 4, 5, 6]
    for i in range(15):
        nums = [(n + i) % 59 + 1 for n in base_numbers]
        # ensure uniqueness
        nums = sorted(set(nums))
        while len(nums) < 6:
            nums.append(max(nums) + 1 if max(nums) < 59 else 1)
        nums = sorted(nums[:6])
        bonus = 59
        date = f"2022-{(i % 12) + 1:02d}-01"
        rows.append(f"{date},{','.join(str(n) for n in nums)},{bonus}")

    csv_path = tmp_path / "draws.csv"
    csv_path.write_text("\n".join(rows) + "\n")

    results = run_pipeline(
        csv_path=str(csv_path),
        epochs=2,
        batch_size=4,
        num_samples=10,
        verbose=False,
    )

    assert "chi_square" in results
    assert results["generated_np"].shape == (10, 6)
    assert results["sim_hits_max"] >= 0
