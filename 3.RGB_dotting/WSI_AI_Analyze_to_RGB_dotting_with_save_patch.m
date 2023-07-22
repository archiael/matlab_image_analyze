%% RGB dotting 스크립트
%% 목적 : WSI 파일(.svs, .tif) 파일을 읽어와서 AI 모델로 조직분류 분석 후 분석결과를 RGB 이미지로 변경하는 스크립트
%% 입력
%% 
% * WSI 파일 목록 (option, 분석할 파일 목록)
% * AI 모델 경로
% * Output 경로
%% 출력
%% 
% * 최대 확률인 클래스 RGB 값으로 변환된 이미지
% * AI가 WSI 이미지를 분석한 결과파일(ovg 형식으로 저장)
%% 1. 환경 세팅

% 작업환경 (메모리) 모두 삭제, 모든 Plot 창 닫기
clear all, close all, format compact, clc

% 사용할 GPU 선택 (1(기본 모니터 출력 및 여러사람 같이 쓰는 용도), 2(분석 전용))
gpuDevice(2);

% 듀얼 GPU로 분석시 아래 변수값 true로 변경
% 분석중인 GPU가 있는지 연락 돌린 후 진행하십시오!
USE_PARALLEL = false;

% subroutines 함수파일을 사용한다면 해당 함수파일(*.m)있는 폴더를 path에 추가
% activateMyNet에 세팅한 함수명을 못찾는다는 에러가 나면
% 아래 addpath()를 사용하여 함수파일이 있는 경로를 아래 () 괄호 사이에 문자열로 넣어주고 실행하십시오.
addpath("");

% AI가 패치를 분석 후 해당 패치를 디스크에 분류하여 저장하도록 하려면
% 아래 
write_fetchImg = false;

% RGB dotting에 사용할 PC 리소스 GPU가 없다면 'auto'나 'cpu'로 변경하십시오
eq_environ = 'gpu';

%% 2. 분석 기준변수 세팅

% WSI 이미지의 border pixel -> 분석결과 이미지 해상도와 관련 있음
border = [12 12];

% AI 분석 시 90도씩 회전하여 4번 분석 후 합산하여 클래스 결정 여부
rotInv = false;

% 클래스 카운팅 제외 클래스
not_count_classes = {'C-15'};

% 0.5um/pixel 이미지로 학습한 모델에 맞게 이미지 리사이징하기위한 기준
nDefaultMPP = 0.5;

% Input WSI file type
input_wsi_type = ".tif"
%% 3. 분석할 WSI 경로

% 분석할 WSI 이미지 폴더 경로
clear dir;
input_wsi_path = uigetdir(pwd,'Select WSI Folder');
disp(input_wsi_path)

allWSIFiles = dir(strcat(input_wsi_path, '/*', input_wsi_type));
disp(strcat("WSI 파일개수 : ", string(numel(allWSIFiles))))

%% 
% 3.1. (Option) 분석할 파일 목록

% 불러올 분석목록 파일을 파일 브라우저에서 선택한다.
% 단, 분석 목록은 파일명(확장자 포함)이 첫번째 항목으로 작성된 엑셀파일이여야 한다.
EXP_LOAD_File = '*.xls;*.xlsx;';
analyze_sample_list_string = [];
str_filter = {EXP_LOAD_File, strcat('Analyze File List(', EXP_LOAD_File, ')');};
title = 'Select Analyze list File';
[filename, cur_path, ~] = uigetfile(str_filter, title, 'MultiSelect', 'off', pwd);

if ~isequal(cur_path,0)
    analyze_sample_list = readtable(fullfile(cur_path, filename));
    
    % 엑셀 헤더 첫번째 항목명 출력
    header_col1_name = string(analyze_sample_list.Properties.VariableNames(1))
    
    % cell을 string 형으로 변환
    analyze_sample_list_string = string(analyze_sample_list.(header_col1_name));
    disp(numel(analyze_sample_list_string));
end

%% 
% 3.2. 출력할 분석결과 엑셀파일 컬럼을 지정한다.

allWSIFiles(1).AppMag = "";
allWSIFiles(1).MPP = "";
allWSIFiles(1).MODE = "";
allWSIFiles(1).slidename = "";

%% 4. VGG19 모델로 학습한 Model 불러오기

% 불러올 파일을 파일 브라우저에서 선택한다.
EXP_LOAD_File = '*.mat;'; % mat(기본 MATLAB 저장방식)
str_filter = {EXP_LOAD_File, strcat('AI Model File(', EXP_LOAD_File, ')');};
title = 'Select AI Model File';
[filename, cur_path, ~] = uigetfile(str_filter, title, 'MultiSelect', 'off', pwd);

load('-mat', fullfile(cur_path,filename));

% 불러온 mat파일이 옳은 AI 모델인지 체크하는 부분 개발해야함

% 불러온 mat파일이 옳은 AI 모델인지 체크하는 부분 개발해야함

% 4.1. AI 모델 마지막 레이어(Classification)에서 클래스 정보와 클래스 이름, 클래스 개수를 가져온다
tissuenames     = unique(myNet.Layers(end).ClassNames);
tissuenames_str = string(unique(myNet.Layers(end).ClassNames));
class_count     = numel(unique(myNet.Layers(end).ClassNames));

% AI 모델 레이어 정보 가져오기
layerNum = numel(myNet.Layers);
outputlayer = myNet.Layers(layerNum, 1);
outputLayerName = outputlayer.Name;

class_num = 1:numel(tissuenames);

% 4.2. AI 모델 첫번째 입력 레이어에서 input 사이즈를 가져온다.
ai_input_layer_size = myNet.Layers(1).InputSize; 

% colors 0~1 사이  double 형식으로 저장되어있다면 RGB 범위인 0~255로 변경한다.
if string(class(colors)) == "double" && sum(sum(colors<=1)) == class_count*3
    colors_uint8 = uint8(colors*255)
end

%% 5. 결과파일 저장 경로 지정

% 분석할 WSI 이미지 폴더 경로
clear dir;
output_file_path = uigetdir(pwd,'Select Output Folder');
disp(output_file_path)

% tif 파일이 있는 경로에 분석결과를 출력할 서브폴더 경로 지정
output_rgb_mat   = strcat(output_file_path, "/mat/");
output_rgb_txt   = strcat(output_file_path, "/txt_소요시간/");
output_rgb_png   = strcat(output_file_path, "/RGB_png/");
output_rgb_patch = strcat(output_file_path, "/patch/");

% 서브폴더 생성
mkdir(output_rgb_mat);
mkdir(output_rgb_txt);
mkdir(output_rgb_png);
mkdir(output_rgb_patch);

% 패치 클래스 서브폴더 생성
for i = 1:numel(tissuenames)
    mkdir(strcat(output_file_path, "/patch/", tissuenames{i}));
end

winopen(output_file_path);

%% 6. 분석할 WSI 파일 수만큼 반복하여 분석

% <분석 파이프라인>
% 6-1. WSI 이미지 MPP 정보로딩
% 6-2. WSI 파일 로딩
%  6-2-1. MPP정보를 활용하여 이미지 resizing
%  6-2-2. 이미지 전처리
%  6-2-3. (New) 전처리한 이미지를 밝기 Normalization
% 6-3. 이미지를 블록단위로 쪼개어 모델을 사용해 분석
% 6-4. 분석결과(확률값)를 이용해 RGB 값으로 치환해 dotting 이미지 저장
% 6-5. 분석결과(확률값)를 이용해 각 조직 개수 계산(카운팅)
i = 1;
for i = 1:numel(allWSIFiles)
    disp(['starting to process next image f=', num2str(i), '--', allWSIFiles(i).name]);

    % 분석 시작시간 측정
    start_t = datetime('now');

    sample_name = strrep(allWSIFiles(i).name, input_wsi_type, '');    
    
    currFilePath = fullfile(input_wsi_path, allWSIFiles(i).name);
    
    % (Option) 엑셀파일을 불러왔다면 현재 파일이 분석대상인지 확인하여 아닌경우 다음 파일을 분석한다.
    if numel(analyze_sample_list_string) > 0
        arr_idx = strcmp(analyze_sample_list_string, allWSIFiles(i).name);
        if ~arr_idx
            disp(strcat(num2str(i), "번째 파일 : ", allWSIFiles(i).name, "은 분석대상이 아니므로 다음파일을 분석합니다."))
            continue;
        end
    end

    % 6-1. MPP정보를 활용하여 이미지 resizing
    curImage_info = imfinfo(currFilePath);
    
    bIsAppMag = false;  % AppMag 정보 유무
    bIsMPP = false;  % MPP정보 유무
    AppMag_split = [];
    nScanImgMPP = 0.5;

    if isfield(curImage_info, 'ImageDescription')
        objInfo_desc = curImage_info.ImageDescription;
        objInfo_desc_split = split(objInfo_desc, "|");
        objInfo_desc_split_str = string(objInfo_desc_split);
        
        for desc_i = 1:numel(objInfo_desc_split_str)
            if(findstr(objInfo_desc_split_str(desc_i), "AppMag") > 0)
                %disp(strcat("Apparent Magnification  : ", objInfo_desc_split_str(desc_i)))
                AppMag_split = split(objInfo_desc_split_str(desc_i), " = ");
                allWSIFiles(i).AppMag = AppMag_split(2);
                bIsAppMag = true;
            elseif(findstr(objInfo_desc_split_str(desc_i), "MPP") > 0)
                %disp(strcat("FIND MPP : ", objInfo_desc_split_str(desc_i)))
                MPP_split = split(objInfo_desc_split_str(desc_i), " = ");
                allWSIFiles(i).MPP = MPP_split(2);
                nScanImgMPP = double(MPP_split(2));
                bIsMPP = true;
            end
        end
    end
    % tif나 svs 파일에 MPP 정보가 없으면 resize 하지 않는다
    if ~bIsMPP
        allWSIFiles(i).MPP = 0.5;
    end
    
    % 6-2. WSI 파일 로딩
    curImage_blocked = blockedImage(currFilePath);
    
    %  6-2-1. MPP정보를 활용하여 이미지 resizing
    if (round(nScanImgMPP, 2) < 0.3 && round(nScanImgMPP, 2) > 0.1)
        nMPP = round(nScanImgMPP / nDefaultMPP, 2);
        curImage_blocked_resize = curImage_blocked.apply(@(bs) imresize(bs.Data, nMPP));
    else
        % resize가 필요없으면 이미지를 불러와서 curImage_org변수에 담는다
        curImage_blocked_resize = curImage_blocked;
    end
    
    %  6-2-2. 패치를 추출하지 않을경우 성능을 위해 전체 WSI 이미지 전처리
    if ~write_fetchImg
        curImage_blocked_resize_grayblur = curImage_blocked_resize.apply(@(bs) imgaussfilt(rgb2gray(bs.Data), 2));
    else
        curImage_blocked_resize_grayblur = curImage_blocked_resize;
    end
    
    %  6-2-3. (New) 전처리한 이미지를 밝기 Normalization
    
    % adapthisteq() 적용을 위해 이미지를 메모리로 로드한다.
    curImage = gather(curImage_blocked_resize_grayblur, "Level", 1);
    
    
    % 분석 시 슬라이딩 간격 조절
    % ex) border [12 12] 세팅 시 block_size [200 200]
    block_size = ai_input_layer_size(1:2) - (2 * border);
    
    % rmov: 결과에서 border에 해당되는 pixel 수 제거용 변수
    rmov = ceil(border(1)/block_size(1));

    activateMyNet = @(I) imageNet_grayScaling_1to3(myNet, I.data, layerNum, outputLayerName, rotInv, eq_environ, sample_name, output_rgb_patch, I.location, write_fetchImg, tissuenames);
    % blockproc() : 큰 TIF 이미지를 224x224x3 이미지 패치 단위로 분석하게 해주는 함수
    mask = blockproc(curImage, block_size, activateMyNet,...
        'TrimBorder',false,'BorderSize',border,...
        'PadPartialBlocks',true,'UseParallel', false, 'DisplayWaitbar', true);
    disp([model_Rev, ' : finished blockproc slice Image output_path']);
    
    % 가장자리를 제거함
    if rmov == 0
        mask_margin = mask((1):(end-rmov),(1):(end-rmov),:); % remove margin
    else
        mask_margin = mask((rmov):(end-rmov),(rmov):(end-rmov),:); % remove margin
    end
    
    mask_size = size(mask_margin);

    % Translation mask(predict value) -> rgbout(RGB value)
    % 확률 1등 클래스의 인덱스를 저장할 매트릭스
    class_idx_1st = zeros(mask_size(1), mask_size(2), 'uint8'); 
    
    % 확률 1등 클래스를 RGB dotting할 값을 세팅할 매트릭스 m x n x 3
    rgbout_1st = zeros(mask_size(1), mask_size(2), 3, 'uint8');
    
    % 확률 1등 클래스를 해당되는 RGB컬러값으로 매핑하여 
    % rgbout_1st 행렬에 저장한다(이후 이미지파일로 저장함)
    mask_margin_1st = zeros(mask_size(1), mask_size(2));
    % curImage_ROI_resize에서 색칠한 색상 영역별 클래스 개수
    count_by_class = zeros(numel(tissuenames));
    
    % 총 pixel 카운트
    cnt_tot = 0;

    for j=1:mask_size(1)
        for k=1:mask_size(2)
            predict_score = sort(mask_margin(j,k,:), 'descend');
            predict_score_1st = predict_score(1);
            mask_margin_1st(j, k) = predict_score_1st;
            
            % RGB 도팅할 1등 확률의 클래스를 구하여 행렬에 저장한다.
            temp_class_num = class_num(mask_margin(j,k,:)==predict_score_1st);
            class_idx_1st(j, k) = temp_class_num(1);
            rgbout_1st(j, k, 1:3) = colors_uint8(class_idx_1st(j,k),:);
            % 2020-09-11 : 카운팅에서 제외할 클래스는 if문에서 체크 continue
            % 모든 클래스를 세는게 더 빠르지만 나중에 계산의 편리를 위해 if continue 처리
            if strcmp(tissuenames{class_idx_1st(j, k)}, not_count_classes{1})
                continue
            end
            cnt_tot = cnt_tot + 1;
            count_by_class(class_idx_1st(j, k)) = count_by_class(class_idx_1st(j, k)) + 1;
        end
    end
    % imshow(rgbout_1st)
    
    % tif 파일 리스트에 slidename 컬럼에 TIF 파일명을 저장함
    allWSIFiles(i).slidename = sample_name;
    
    % 각 클래스별 pixel 카운트 숫자 
    for m=1:numel(tissuenames)
        allWSIFiles(i).(strrep(string(tissuenames(m)), '-', '_')) = count_by_class(m);
    end
    allWSIFiles(i).tot_class = cnt_tot;
            
    % 분석결과 이미지를 png 파일로 저장함
    target_file = strcat(output_rgb_png,'/', sample_name, '_', model_Rev, '_border_',num2str(border(1)),'.png');
    imwrite(rgbout_1st(:,:,1:3),target_file);
    
    % .mat 파일로 분석결과를 저장함
    save(strcat(output_rgb_mat,'/', sample_name, '_', model_Rev, '_analyze_border_', num2str(border(1)),'.mat'), 'mask_margin', 'model_Rev', 'tissuenames', 'colors', 'border', 'class_idx_1st', 'rgbout_1st', 'mask_margin_1st');

    % 분석이 끝난 시간을 기록함
    % 분석 종료시간 측정
    output_T = cell2table({sample_name; currFilePath});
    end_t = datetime('now');
    duration_t = end_t - start_t;
    duration_t.Format = 'm';
    
    save_path = strcat(output_rgb_txt, '/', sample_name,'_', model_Rev, '_분석종료_', string(duration_t), '.txt');
    writetable(output_T, save_path);
    
end
% TIF 리스트를 테이블 형태로 변환하여 엑셀파일로 저장한다.
save_allFiles = struct2table(allWSIFiles);
strToday = datestr(now,'yyyymmdd_HHMM');
save_path = strcat(output_file_path, '/', model_Rev,'_클래스분류_카운트', strToday, '.xlsx');
writetable(save_allFiles, save_path);