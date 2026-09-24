from traffic.counter import LineCounter

A, B = (0, 100), (200, 100)  # vach ngang y=100


def run(counter, path, tid=1, cls="car"):
    events = []
    for i, p in enumerate(path):
        events += counter.update([(tid, cls, p)], i)
    return events


def test_crossing_down_counts_in():
    c = LineCounter(A, B)
    ev = run(c, [(50, 80), (50, 95), (50, 110)])
    assert len(ev) == 1 and ev[0].direction == "in" and ev[0].class_name == "car"
    assert c.total == 1


def test_crossing_up_counts_out():
    c = LineCounter(A, B)
    ev = run(c, [(50, 120), (50, 90)])
    assert ev[0].direction == "out"
    assert c.counts["out"]["car"] == 1


def test_no_double_count_when_jittering_on_line():
    c = LineCounter(A, B)
    run(c, [(50, 90), (50, 110), (50, 90), (50, 110), (50, 90)])
    assert c.total == 1


def test_no_count_outside_segment():
    c = LineCounter(A, B)
    run(c, [(300, 80), (300, 120)])
    assert c.total == 0


def test_no_count_without_crossing():
    c = LineCounter(A, B)
    run(c, [(50, 10), (60, 50), (70, 90)])
    assert c.total == 0


def test_majority_vote_class():
    c = LineCounter(A, B)
    c.update([(7, "truck", (50, 60))], 0)
    c.update([(7, "truck", (50, 70))], 1)
    c.update([(7, "car", (50, 80))], 2)
    ev = c.update([(7, "truck", (50, 120))], 3)
    assert ev[0].class_name == "truck"


def test_multiple_tracks_independent():
    c = LineCounter(A, B)
    c.update([(1, "car", (20, 90)), (2, "motorbike", (80, 110))], 0)
    c.update([(1, "car", (20, 110)), (2, "motorbike", (80, 90))], 1)
    assert c.counts["in"]["car"] == 1
    assert c.counts["out"]["motorbike"] == 1
    assert c.by_class() == {"car": 1, "motorbike": 1}
