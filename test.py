# x = {
#         "id": 1 ,
#         "data": [
#             {
#                 "track_id": "2983hawdjjawd",
#                 "listen": 3
#             },
#             {
#                 "track_id": "23671323azcxxc",
#                 "listen": 13
#             },
#         ]
#     }

# s = [i['listen'] for i in x['data']]

# print(sum(s))

# import numpy as np

# arr = np.array([[3, 5, 7, 9, 11],[2, 4, 6, 8, 10]])
# arr2 = arr[:,:2]

# print(arr2)
# print("-----------")

# arr = np.array([[[3, 5, 7, 9, 11],
#                  [2, 4, 6, 8, 10]],
#                 [[5, 7, 8, 9, 2],
#                  [7, 2, 3, 6, 7]]])
# arr2 = arr[1:]

# print(arr2)

from datetime import datetime

def get_data_date(timestamp: str) -> datetime:
    datetime_timestamp = datetime.strptime(timestamp, f'%Y-%d-%m')

    return datetime_timestamp

x = get_data_date("2023-12-10")

print(x.month)
print(type(x.month))
print(x.day)
print(type(x.day))
