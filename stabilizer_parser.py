def parse_to_tuple(matrix, n): #n is the n in [[n,k,d]]
    arr = matrix.split("\n")
    x_arr = []
    z_arr = []
    cx_arr = []
    for x in arr:
        x_temp = []
        z_temp = []
        cx_temp = []
        splits = x.split("|")
        for i in range(len(splits)): #2 times
            side = splits[i]
            nums = side.split(" ")
            print("nums", nums)
            print(len(nums))
            print(n)
            assert(len(nums) == n)
            for j in range(len(nums)):
                number = nums[j]
                number = number.replace("[", "")
                number = number.replace("]", "")
                #print("number", number)
                if (int(number) == 1):
                    if (i == 0):
                        x_temp.append(j)
                    else:
                        assert(i == 1)
                        z_temp.append(j)
                    if (j not in cx_temp):
                        cx_temp.append(j)
        #print("x temp", x_temp)
        x_arr.append(x_temp)
        z_arr.append(z_temp)
        cx_arr.append(cx_temp)

    

    return (x_arr, z_arr, cx_arr, n)

def greedy_parallel_schedule(task_lists):
    schedule = []  # List of parallel batches, each is a list of task indices
    task_sets = [set(task) for task in task_lists]

    for i, task in enumerate(task_sets):
        placed = False
        for batch in schedule:
            # Check if task conflicts with any task in this batch
            if all(task.isdisjoint(task_sets[j]) for j in batch):
                batch.append(i)
                placed = True
                break
        if not placed:
            schedule.append([i])

    # Total time = sum of max lengths of tasks per time slot
    total_time = sum(max(len(task_lists[i]) for i in batch) for batch in schedule)

    return schedule, total_time


def real_greedy_parallel_schedule(x_arr, z_arr):
    schedule = []
    x_sets = [set(x) for x in x_arr]
    z_sets = [set(z) for z in z_arr]

    for i in range(len(x_arr)):
        placed = False
        for batch in schedule:
            valid = True
            for j in batch:
                # Direct conflicts
                x_conflict = not x_sets[i].isdisjoint(x_sets[j])
                z_conflict = not z_sets[i].isdisjoint(z_sets[j])

                # Cross-type conflicts
                xz_conflict = not x_sets[i].isdisjoint(z_sets[j])
                zx_conflict = not z_sets[i].isdisjoint(x_sets[j])

                # If more than one type of conflict occurs, it's invalid
                conflict_types = sum([
                    x_conflict,
                    z_conflict,
                    xz_conflict,
                    zx_conflict
                ])

                if conflict_types > 1:
                    valid = False
                    break
            if valid:
                batch.append(i)
                placed = True
                break
        if not placed:
            schedule.append([i])

    total_time = sum(
        max(len(x_arr[i]) + len(z_arr[i]) for i in batch) for batch in schedule
    )

    return schedule, total_time




"""
def count_ones_in_string(matrix_string):
    return sum(1 for token in matrix_string.split() if token == '1')
"""

def count_gates(cx_arr):
    total = 0
    for x in cx_arr:
        for g in x:
            total += 1
    return total

matrix = """[1 0 0 1 1 0 0 1 1 0 0 0 1 0 1|0 0 1 1 0 0 1 0 1 1 0 1 0 1 1]
[0 0 1 1 0 0 1 0 1 1 0 1 0 1 1|1 0 1 0 1 0 1 1 0 1 0 1 1 1 0]
[0 1 1 1 1 0 0 0 0 0 0 1 1 1 1|0 0 0 0 0 0 0 1 1 1 1 1 1 0 0]
[0 0 0 0 0 0 0 1 1 1 1 1 1 0 0|0 1 1 1 1 0 0 1 1 1 1 0 0 1 1]
[0 0 0 0 0 1 1 1 1 1 1 1 1 1 1|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0|0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]"""

x_arr, z_arr, cx_arr, _ = parse_to_tuple(matrix, 15)
print(x_arr)
print(z_arr)
schedule, time = real_greedy_parallel_schedule(x_arr, z_arr)
print("Schedule", schedule)
print("Parallel time steps taken", time)
print("Serial time steps", count_gates(cx_arr))