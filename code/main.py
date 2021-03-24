import pandas as pd
from feature_generation import FeatureGeneration as FG


def main():

    # def find_count(m, n, mid):
    #     count = 0
    #     for i in range(1, n + 1):
    #         if mid // i >= m:
    #             count += m
    #         else:
    #             count += mid // i
    #     return count
    #
    # n, m, k = map(int, input().split(" "))
    # min = 1
    # max = n * m
    # while min <= max:
    #     mid = (min + max) // 2
    #     count = find_count(m, n, mid)
    #     if count < k:
    #         min = mid + 1
    #     else:
    #         max = mid - 1
    # print(min)

    a,b,c = map(int,input().split(' '))



if __name__ == "__main__":
    main()
