import difflib

def check_diff(file1, file2):
    d = difflib.Differ()
    with open(file1) as f1, open(file2) as f2:
        diff = d.compare(f1.readlines(), f2.readlines())
        print(''.join(diff))

# check_diff('test_data/19-145_ref.txt', 'results/19-145.txt')
# check_diff('test_data/19-110_ref.txt', 'results/19-110.txt')
check_diff('test_data/19-126_ref.txt', 'results/19-126.txt')