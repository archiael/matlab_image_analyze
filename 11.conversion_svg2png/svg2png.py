
import os
import sys

# linux 
#input=sys.argv[1]
#output=sys.argv[1][:-4]+'.png'

# window
path = sys.argv[1]
#path = "Z:\\2023_01_NPOG확증임상\\LEICA\\02 무작위추출 105\\99.TEMP\\20230425 IoU between p1 and p2\\ROI p1\\re"
#path = "Z:\\2023_01_NPOG확증임상\\LEICA\\02 무작위추출 105\\99.TEMP\\20230503 IoU\\p1"

file_ext = r'.svg'
file_list = [file for file in os.listdir(path) if file.endswith(file_ext)]
print(file_list)

if len(file_list) > 0:
    try:
        os.mkdir((path+"\\png\\"))
    except:
        print("png mkdir")


# 2023-04-26
# renderPM의 투명도에 따라 줄이 생기는 문제로 cairosvg 패키지로 변경함
#from svglib.svglib import svg2rlg
#from reportlab.graphics import renderPM
import cairosvg

for idx, file_name in file_list:
    target_name = path + "\\png\\" + file_name.replace('.svg', '.png')
    print(idx + " step : " + target_name)
    # rederPM 패키지용 코드
    #drawing = svg2rlg(path+"\\"+file_name)
    #renderPM.drawToFile(drawing, target_name, fmt="PNG")
    # cairosvg 패키지용 코드
    cairosvg.svg2png(url=(path+"\\"+file_name), write_to=(target_name))
