# # # egi_to_1020 = {
# # #     'E3': 'AF4',
# # #     'E4': 'F2',
# # #     'E6': 'FCZ',
# # #     'E9': 'FP2',
# # #     'E11': 'FZ',
# # #     'E13': 'FC1',
# # #     'E15': 'FPZ',
# # #     'E19': 'F1',
# # #     'E22': 'FP1',
# # #     'E23': 'AF3',
# # #     'E24': 'F3',
# # #     'E27': 'F5',
# # #     'E28': 'FC5',
# # #     'E29': 'FC3',
# # #     'E30': 'C1',
# # #     'E33': 'F7',
# # #     'E34': 'FT7',
# # #     'E36': 'C3',
# # #     'E37': 'CP1',
# # #     'E41': 'C5',
# # #     'E42': 'CP3',
# # #     'E45': 'T7',
# # #     'E46': 'TP7',
# # #     'E47': 'CP5',
# # #     'E51': 'P5',
# # #     'E52': 'P3',
# # #     'E55': 'CPZ',
# # #     'E60': 'P1',
# # #     'E62': 'PZ',
# # #     'E65': 'PO7',
# # #     'E67': 'PO3',
# # #     'E70': 'O1',
# # #     'E72': 'POZ',
# # #     'E75': 'OZ',
# # #     'E77': 'PO4',
# # #     'E83': 'O2',
# # #     'E85': 'P2',
# # #     'E87': 'CP2',
# # #     'E90': 'PO8',
# # #     'E92': 'P4',
# # #     'E93': 'CP4',
# # #     'E97': 'P6',
# # #     'E98': 'CP6',
# # #     'E102': 'TP8',
# # #     'E103': 'C6',
# # #     'E104': 'C4',
# # #     'E105': 'C2',
# # #     'E111': 'FC4',
# # #     'E112': 'FC2',
# # #     'E116': 'FT8',
# # #     'E117': 'FC6',
# # #     'E122': 'F8',
# # #     'E123': 'F6',
# # #     'E124': 'F4'
# # # }

# # # CHANNEL_DICT = ['FP1', 'FPZ', 'FP2',
# # #                 'AF3', 'AF4',
# # #                 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
# # #                                                     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
# # #                                                     'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
# # #                                                     'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
# # #                                                     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
# # #                                                     'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
# # #                                                     'O1', 'OZ', 'O2', ]

# # # print(len(CHANNEL_DICT))


# # # keys = list(egi_to_1020.keys())

# # # print("length of keys: ", len(keys))

# # # entire = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E39', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64',
# # #           'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124']

# # # remove = []
# # # for ch in entire:
# # #     if ch not in keys:
# # #         remove.append(ch)


# # # print(remove)
# # # values = list(egi_to_1020.values())

# # # print("values: ", values)
# # # print("length of values: ", len(values))

# # # absentchannels = []
# # # for ch in CHANNEL_DICT:
# # #     print("ch: ", ch)
# # #     if ch not in values:
# # #         absentchannels.append(ch)

# # # print("absent channels: ", absentchannels)

# # # # # use_channels_names = ['FP1', 'FPZ', 'FP2',
# # # # #                       "AF7", 'AF3', 'AF4', "AF8",
# # # # #                       'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
# # # # #                                            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
# # # # #                                            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
# # # # #                                            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
# # # # #                                            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
# # # # #                                            'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8',
# # # # #                                            'O1', 'OZ', 'O2', ]

# # # # # use_channels_names = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
# # # # #                       'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

# # # # # ch_names = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
# # # # #             'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

# # # # # use_channels_names = ['PZ', 'C2', 'P5', 'P6', 'TP8', 'C5', 'FC4', 'FT7', 'AF4', 'POZ', 'F6', 'TP7', 'PO7', 'PO4', 'O2', 'F8', 'F4', 'T7', 'CP6', 'PO8', 'C3', 'CP1', 'CP4', 'F3', 'OZ', 'FC3',
# # # # #                       'FT8', 'F7', 'FP2', 'PO3', 'P4', 'F5', 'FC2', 'P2', 'AF3', 'CPZ', 'F2', 'CP5', 'FP1', 'FC1', 'P1', 'FZ', 'FPZ', 'CP3', 'O1', 'P3', 'C6', 'FC6', 'C4', 'F1', 'CP2', 'FCZ', 'FC5', 'C1']

# # # # # ch_names = ['PZ', 'C2', 'P5', 'P6', 'TP8', 'C5', 'FC4', 'FT7', 'AF4', 'POZ', 'F6', 'TP7', 'PO7', 'PO4', 'O2', 'F8', 'F4', 'T7', 'CP6', 'PO8', 'C3', 'CP1', 'CP4', 'F3', 'OZ', 'FC3',
# # # # #             'FT8', 'F7', 'FP2', 'PO3', 'P4', 'F5', 'FC2', 'P2', 'AF3', 'CPZ', 'F2', 'CP5', 'FP1', 'FC1', 'P1', 'FZ', 'FPZ', 'CP3', 'O1', 'P3', 'C6', 'FC6', 'C4', 'F1', 'CP2', 'FCZ', 'FC5', 'C1']

# # # # use_channels_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
# # # #                       'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
# # # # ch_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2',
# # # #             'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']


# # orderone = ['AF4', 'F2', 'FCZ', 'FP2', 'FZ', 'FC1', 'FPZ', 'F1', 'FP1', 'AF3', 'F3', 'F5', 'FC5', 'FC3', 'C1', 'F7', 'FT7', 'C3', 'CP1', 'C5', 'CP3', 'T7', 'TP7', 'CP5', 'P5',
# #             'P3', 'CPZ', 'P1', 'PZ', 'PO7', 'PO3', 'O1', 'POZ', 'OZ', 'PO4', 'O2', 'P2', 'CP2', 'PO8', 'P4', 'CP4', 'P6', 'CP6', 'TP8', 'C6', 'C4', 'C2', 'FC4', 'FC2', 'FT8', 'FC6', 'F8', 'F6', 'F4']

# # ordertwo = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
# #             'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

# # for ch in orderone:
# #     if ch not in ordertwo:
# #         print(f"Channel {ch} is in orderone but not in ordertwo.")


# ordering = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#             'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
#             'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6',
#             'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 'O1', 'OZ', 'O2', ]

# best = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
#         'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

# print(set(ordering) - set(best))
# print(set(best) - set(ordering))

# # adapted version to test
# egi_to_1020 = {
#     'E3': 'AF4',
#     'E4': 'F2',
#     'E6': 'FCZ',
#     'E9': 'FP2',
#     'E11': 'FZ',
#     'E13': 'FC1',
#     'E15': 'FPZ',
#     'E19': 'F1',
#     'E23': 'AF3',
#     'E24': 'F3',
#     'E28': 'FC5',
#     'E29': 'FC3',
#     'E30': 'C1',
#     'E33': 'F7',
#     'E34': 'FT7',
#     'E36': 'C3',
#     'E37': 'CP1',
#     'E41': 'C5',
#     'E42': 'CP3',
#     'E45': 'T7',
#     'E46': 'TP7',
#     'E47': 'CP5',
#     'E51': 'P5',
#     'E52': 'P3',
#     'E55': 'CPZ',
#     'E58': 'P7',
#     'E60': 'P1',
#     'E62': 'PZ',
#     'E65': 'PO7',
#     'E67': 'PO3',
#     'E70': 'O1',
#     'E72': 'POZ',
#     'E75': 'OZ',
#     'E77': 'PO4',
#     'E83': 'O2',
#     'E85': 'P2',
#     'E87': 'CP2',
#     'E90': 'PO8',
#     'E92': 'P4',
#     'E93': 'CP4',
#     'E97': 'P6',
#     'E96': 'P8',
#     'E98': 'CP6',
#     'E102': 'TP8',
#     'E103': 'C6',
#     'E104': 'C4',
#     'E105': 'C2',
#     'E108': 'T8',
#     'E111': 'FC4',
#     'E112': 'FC2',
#     'E116': 'FT8',
#     'E117': 'FC6',
#     'E122': 'F8',
#     'E124': 'F4'
# }

orderfour = ['F3', 'FP2', 'FT7', 'FC4', 'FZ', 'P5', 'FC2', 'C3', 'F8', 'F2', 'C6', 'AF4', 'P1', 'CPZ', 'FP1', 'FC1', 'P2', 'F4', 'PO8', 'FC5', 'OZ', 'C4', 'TP7', 'PZ', 'PO3', 'F7',
             'F6', 'FPZ', 'FC3', 'CP2', 'PO7', 'F5', 'P4', 'PO4', 'CP5', 'O2', 'FC6', 'C1', 'CP3', 'C2', 'CP1', 'TP8', 'FCZ', 'T7', 'P3', 'C5', 'CP4', 'P6', 'O1', 'AF3', 'FT8', 'CP6', 'POZ', 'F1']

orderone = ['AF4', 'F2', 'FCZ', 'FP2', 'FZ', 'FC1', 'FPZ', 'F1', 'FP1', 'AF3', 'F3', 'F5', 'FC5', 'FC3', 'C1', 'F7', 'FT7', 'C3', 'CP1', 'C5', 'CP3', 'T7', 'TP7', 'CP5', 'P5',
            'P3', 'CPZ', 'P1', 'PZ', 'PO7', 'PO3', 'O1', 'POZ', 'OZ', 'PO4', 'O2', 'P2', 'CP2', 'PO8', 'P4', 'CP4', 'P6', 'CP6', 'TP8', 'C6', 'C4', 'C2', 'FC4', 'FC2', 'FT8', 'FC6', 'F8', 'F6', 'F4']

print(set(orderone) - set(orderfour))
