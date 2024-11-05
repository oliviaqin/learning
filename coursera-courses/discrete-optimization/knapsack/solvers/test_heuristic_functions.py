from heuristic_functions import HeuristicFunctions
from utils.constants import Item

TEST_ITEMS = [Item(0, 45, 5), Item(1, 48, 8), Item(2, 35, 3)]
TEST_CAPACITY = 10

def test_heuristic_functions():
    print(f"optimistic total is: {HeuristicFunctions.optimistic_total(TEST_ITEMS)}")
    assert HeuristicFunctions.optimistic_total(TEST_ITEMS) == 128
    print(f"belgian chocolate total is: {HeuristicFunctions.belgian_chocolate(TEST_ITEMS, TEST_CAPACITY)}")
    assert HeuristicFunctions.belgian_chocolate(TEST_ITEMS, TEST_CAPACITY) == 92

if __name__ == '__main__':
    test_heuristic_functions()