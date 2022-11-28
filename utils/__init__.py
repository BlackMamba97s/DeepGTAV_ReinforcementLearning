import time


def list_equal(list1, list2):
    _list1 = list1.copy()
    _list2 = list2.copy()
    _list1.sort()
    _list2.sort()
    return _list1 == _list2


def count_down(number):
    for i in range(number)[::-1]:
        print(i)
        time.sleep(1)