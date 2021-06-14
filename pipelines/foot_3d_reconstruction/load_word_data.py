
import numpy as np
import docx


def load_docx(path, foot):
    document = docx.Document(path)  # 读入文件
    tables = document.tables  # 获取文件中的表格集

    table = tables[0]  # 获取文件中的第9个表格
    
    item_description = []
    data_list = []
    
    for i in [5, 6, 7, 8, 9, 10, 16, 17, 18]:  # 从表格第六行开始循环读取表格数据
            proNum = table.cell(i, 1).text  # 项目描述

            # print(f"第{i}行：")
            # print(f"项目描述：{proNum}")            
            if foot == 'left':
                foot = table.cell(i, 3).text  # 左脚
                # print(f"左脚：{leftFoot}")
            else:
                foot = table.cell(i, 5).text   # 右脚
                # print(f"右脚：{rightFoot}")
            item_description.append(proNum)
            data_list.append(eval(foot))


    return item_description, np.asarray(data_list, dtype=np.float32)
