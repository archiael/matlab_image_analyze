% 2023-04-12 두개의 서로 다른 ROI 영역을 합쳐서 합집합인 영역을 png 파일로 저장
% <Input>
% 1. (필수) ROI 1 이미지 파일(png) * SVG인 경우 파이썬 SVG to PNG 를 이용하여 변환
% 2. (필수) ROI 2 이미지 파일(png) * SVG인 경우 파이썬 SVG to PNG 를 이용하여 변환
% ps. SVG를 PNG로 변환하는 방법은 11.conversion_svg2png폴더의 svg2png사용법.txt 참조

clear all, close all, format compact, clc

rng('default');
rng(2);

% sigma 별로 엑셀파일을 출력하기위해 초기화
clearvars allSummaryROIFiles;

% output 파일 경로
% exsample output 경로
result_path = "E:\git\matlab_source_org\AI_model\10.Dice_IoU\2.result(example)\retulst_Merge_ROI1_and_ROI2";

% 실제경로 탐색기로 열기
mkdir(result_path)
winopen(result_path);
% imread에서 png 파일 폴더경로를 환경변수에 추가
addpath(result_path);


%% DICE나 IoU를 계산할 ROI 파일 목록 불러오기 (필수)
% ROI 1 세트 목록 = DICE&IoU 계산할 샘플목록
clear dir;
input_ROI1_Path = "E:\git\matlab_source_org\AI_model\10.Dice_IoU\1.input(example)\ROI1\";
list_ROI1_Files = dir(strcat(input_ROI1_Path,'/*.png'));
disp(strcat("ROI 파일 ", num2str(numel(list_ROI1_Files)), "개 로딩"))

% ROI 2 세트 목록 = DICE&IoU 계산할 샘플목록
clear dir;
input_ROI2_Path = "E:\git\matlab_source_org\AI_model\10.Dice_IoU\1.input(example)\ROI2\";

list_ROI2_Files = dir(strcat(input_ROI2_Path,'/*.png'));
disp(strcat("ROI 파일 ", num2str(numel(list_ROI2_Files)), "개 로딩"))

clearvars allSummaryROIFiles;
allSummaryROIFiles = struct([]);

file_i = 1;

for file_i = 1:numel(list_ROI1_Files)
    disp(strcat("ROI1 file_i : ", num2str(file_i), "  /  sample name : ", list_ROI1_Files(file_i).name));
    disp(strcat("ROI2 file_i : ", num2str(file_i), "  /  sample name : ", list_ROI2_Files(file_i).name));
    %% ROI 1과 ROI2를 불러와서 DICE와 IoU를 구하기 위한 전처리를 진행
    % 엑셀로 출력할 변수에 파일 이름 세팅
    allSummaryROIFiles(file_i).BW1_file = list_ROI1_Files(file_i).name;
    allSummaryROIFiles(file_i).BW2_file = list_ROI2_Files(file_i).name;
    
    % 1번 ROI파일 불러오기
    [BW1, map, alpha] = imread(fullfile(list_ROI1_Files(file_i).folder, list_ROI1_Files(file_i).name));
    
    % 1번 ROI 색상을 흑/백(Tumor)로 변경하여 리사이징
    BW1_gray = im2gray(BW1);

    % 1번 ROI 영역을 True 값으로 변환
    BW1_binary = imbinarize(BW1_gray, 'global');
    
    % 2번 ROI 파일 불러오기
    [BW2, map, alpha] = imread(fullfile(list_ROI2_Files(file_i).folder, list_ROI2_Files(file_i).name));
    
    % ROI 색상을 흑/백(Tumor)로 변경하여 리사이징
    BW2_gray = im2gray(BW2);
    
    % ROI 영역을 True 값으로 변환
    BW2_binary = imbinarize(BW2_gray, 'global');
    
    % ROI 영역을 Mergeg
    % ROI1과 ROI2를 병합한다.
    BW1_BW2_merge = BW1_binary | BW2_binary;
    
    %% 테스트용 Merge 과정 출력
    figure("Position", [0 0 1920 1024])
    % 2행 3열 plot 첫번째 plot으로 BW1 이미지 출력
    subplot(2,4,1); imshow(BW1);
    title("ROI 1")

    % 2행 3열 plot 두번째 plot으로 BW1_gray 이미지 출력
    subplot(2,4,2); imshow(BW1_gray);
    title("ROI 1-grayscale")

    % 2행 3열 plot 세번째 plot으로 BW1_binary 이미지 출력
    subplot(2,4,3); imshow(BW1_binary);
    title("ROI 1-Binary")

    % 2행 3열 plot 네번째 plot으로 BW2 이미지 출력
    subplot(2,4,5); imshow(BW2);
    title("ROI 2")

    % 2행 3열 plot 다섯번째 plot으로 BW2_gray 이미지 출력
    subplot(2,4,6); imshow(BW2_gray);
    title("ROI 2-grayscale")

    % 2행 3열 plot 여섯번째 plot으로 BW2_binary 이미지 출력
    subplot(2,4,7); imshow(BW2_binary);
    title("ROI 2-Binary")
    
    % ROI 영역을 imfuse하여 겹쳐서 이미지로 저장한다.
    img_imfuse = imfuse(BW1_binary, BW2_binary);
    subplot(2,4,4); imshow(img_imfuse);
    title("ROI 1, 2 - imfuse")

    % 2행 4열 plot 1행+2행 4번째 열 plot으로 BW1_BW2_merge 이미지 출력
    subplot(2,4,8); imshow(BW1_BW2_merge);
    title("ROI 1, 2 - Merged")

    % figure를 저장하려면 print()함수를 사용 가능함
%     print(gcf,'-djpeg','-r300', strcat(result_path,'/', list_ROI1_Files(file_i).name, '_', 'imfuse', '_DICE_IoU.jpg'));
%     close all;

    % png 파일로 저장
    imwrite(BW1_BW2_merge, strcat(result_path,'/', replace(list_ROI1_Files(file_i).name, ".png", ""), '_', 'Merge.png'));

    close all;
    



    %% 전처리한 ROI1과 ROI2를 dice()와 jaccard()를 이용하여 DICE와 IoU 계수를 구한다.
    % DICE 계수 구하기
    DICE_ROI_tumor = dice(BW1_binary, BW2_binary);
    % 출력할 변수에 dice 계수 기록
    allSummaryROIFiles(file_i).DICE_ROI_tumor = DICE_ROI_tumor;
    
    % IoU 구하기
    IoU_ROI_tumor = jaccard(BW1_binary, BW2_binary);
    % 출력할 변수에 IoU 계수 기록
    allSummaryROIFiles(file_i).IoU_ROI_tumor = IoU_ROI_tumor;

end

%% 계산한 DICE와 IoU 모두 엑셀파일로 출력
% 필터 크기별로 엑셀파일을 출력한다.
strToday = datestr(now,'yyyymmdd_HHMM');
save_allSummaryROIFiles = struct2table(allSummaryROIFiles);
writetable(save_allSummaryROIFiles, strcat(result_path, '/', num2str(numel(allSummaryROIFiles)), '개샘플_ROI별_DICE_IoU', '_', strToday,'.xlsx'));