%% 2023-01-30 기존 DICE를 dice()로 IoU를 jaccard()로 MATLAB 함수를 이용하여 계산하는 스크립트 개발
% <Input>
% 1. (필수) 분석할 샘플 이름 목록 엑셀파일(엑셀형식)
% 2. (필수) AI 모델(mat)
% 3. (필수) ROI 이미지 파일(png)
% 4. (필수) AI 분석완료 파일(mat)
% 5. (옵션) H&E mini 사이즈 이미지

% <Output>
% 1. 샘플별 Plot 1장 (jpg)
% 2. 샘플별 Cutoff별(0~1, 0.1텀) ROI 표시 이미지 11장 (jpg)
% 3. 필터 크기별(arr_size_filter) 엑셀파일 (xlsx), Input 1번의 샘플개수 * cutoff 개수 만큼의 행이 생성

clear all, close all, format compact, clc

rng('default');
rng(2);

bShowPlot = true;

gpuDevice(2);

addpath("E:\git\AI_model_kimhyuntae\8.analysis_by_ROI")

%% 분석에 필요한 딥러닝 모델 로딩
% 모델파일이 있는 폴더 경로
model_path = 'Z:\DEEP_LEARNING_RGB_dotting\2.trained_model\Rev0.01.xx_SET_002_stomach_cancer_class_define\Rev0.01.13_LEICA_C04C05';
% 모델파일 명
model_file = 'tuning_C04C05_class_15_Rev0.01.13_LEICA_C04C05.mat';

% newData1 변수에 모델파일 로딩
newData1 = load('-mat', fullfile(model_path, model_file));

% newData1 struct 변수의 field로 접근할 수 있는 변수를
% 개별변수 형태로 메모리로 로딩
vars = fieldnames(newData1);
for model_i = 1:length(vars)
    assignin('base', vars{model_i}, newData1.(vars{model_i}));
end

% 2023-01-30 / 김현태 : 기존 colors 값은 0~1값인것을 원래 0~255 값으로 치환
colors = colors*255;

% myNet에 저장한 모델에서 Input 이미지 크기를 가져온다.
% VGG19모델은 기본 224x224x3 크기임
layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize;

clearvars testing_tbl training_categories training_tbl vars newData1;

%% 분석에 필요한 변수 세팅

% str_Set_Name = "TCGA.STAD";
% str_Set_Name = "SET_003_tuning";
% str_Set_Name = "SET_003_test";
% str_Set_Name = "SET_003_tuning_보고서용";
% str_Set_Name = "SET_003_test_보고서용";
str_Set_Name = "nPOG_확증임상";

if strcmp(str_Set_Name, "TCGA.STAD")
    str_Excel_File_Path = "analysis_for_DICE_IoU_by_ROI_tcga_346ea.xlsx";
    input_resize6_png_Path = 'Z:\SET_013_TCGA\TCGA.STAD\2.resize_6%';
    input_ROI_Path = 'Z:\SET_013_TCGA\TCGA.STAD\5.ROI\2.analysis\20220811';
    input_mat_Path = 'Z:\SET_013_TCGA\TCGA.STAD\3.RGB_dotting\Rev0.01.13_LEICA_C04C05\mat';

elseif strcmp(str_Set_Name, "colon_200_leica")
    str_Excel_File_Path = "analysis_for_DICE_IoU_by_ROI_colon_200ea.xlsx";
    input_resize6_png_Path = 'Z:\SET_010_Colon200_Leica\2.resize_4%';
    input_ROI_Path = 'Z:\SET_010_Colon200_Leica\5.ROI\20220719_PAIK_CRC\png\';
    input_mat_Path = 'Z:\SET_010_Colon200_Leica\3.RGB_dotting\Rev0.01.13_LEICA_C04C05\mat\';
else % 기타 샘플은 아래 변수로 처리
    str_Excel_File_Path = "analysis_for_DICE_IoU_by_ROI_nPOG_70ea.xlsx";
    input_ROI_Path = 'Z:\2023_01_NPOG확증임상\LEICA\9.올리바지 확증임상 AT2장비 스캔_신촌세브란스(140개)_20230104\5.ROI\01 EJJang\png\';
    input_resize6_png_Path = 'Z:\2023_01_NPOG확증임상\LEICA\9.올리바지 확증임상 AT2장비 스캔_신촌세브란스(140개)_20230104\2.resize_5%';
    input_mat_Path = 'Z:\2023_01_NPOG확증임상\LEICA\9.올리바지 확증임상 AT2장비 스캔_신촌세브란스(140개)_20230104\3.RGB_dotting\mat\';
end


%% DICE & IoU 계산한 결과를 저장할 폴더 지정
output_path = strcat(input_ROI_Path, '\Jaccard_ROI바이너리화_계수_보고서_', model_Rev);
mkdir(output_path);
winopen(output_path);

% 최종 분류 카테고리 : 병변, 비병변 Other 3개로 나누어 카운트 진행
tissuenames_category = {'Tumor', 'Normal', 'Others', 'Back'};

% 분석할 샘플 목록 엑셀파일을 읽어온다.
all_sample_list = readtable(str_Excel_File_Path);

% 엑셀 헤더 정보 출력
disp(all_sample_list.Properties.VariableNames(:));

% 헤더가 몇개인지 출력
disp(numel(all_sample_list.Properties.VariableNames));

% cell을 string 형으로 변환
all_sample_list_string = string(all_sample_list.filename(:));

% 샘플 개수를 명령창에 출력
disp(numel(all_sample_list_string));

% str_filter_func : imfilter(평균필터) / imgaussfilt(가우스 평활화 필터)
str_filter_func = "imfilter";
% arr_size_filter = [2]; % 가우시안 필터 조건식 => 2*ceil(2*sigma)+1
arr_size_filter = [9]; % 평균 필터 조건  9px일때 1mm에 근접

% 필터 후 cutoff 쓰레스홀드 수치별 비교용
% 최적 cutoff 수치는 0.5~0.6
% arr_cutoff_max = [0:0.1:1];
% 2022-10-04 김수정 : 스크립트 개발 중 너무 많은 출력을 방지하고자 최적의 컷오프로 고정함
% 추 후 컷오프별로 테스트하려면 117라인 주석을 풀어서 사용
% arr_cutoff_max = [0.5];
arr_cutoff_max = [128];

% 2022-07-19 ROI 색상은 이제 1가지만 되었으니 ROI 색상 변수를 반복할 green cutoff용으로 지정
% cutoff_color_name : 반복할 cutoff 개수를 지정
cutoff_color_name = string(arr_cutoff_max);
arr_cutoff_max_cnt = numel(arr_cutoff_max);

% cutoff_color_RGB : 반복할 cutoff 개수 만큼 색상을 지정
% Qu-Path 'tumor' 지정 ROI 색상
% 예외적용 : 'tumor' 지정 안할 시 ROI 색상은 [238, 178, 178]
cutoff_color_RGB = [255 178 178;];
cutoff_color_RGB_repmat = repmat(cutoff_color_RGB, numel(cutoff_color_name), 1);

% ROI의 색상이 여러가지면 아래 숫자를 1에서 해당 색생개수로 변경해야함
cutoff_color_cnt = 1;

%% 슬라이드 H&E tif  이미지 목록 불러오기 (Option)
% 첫번째 main 세트
clear dir;
allPNGFiles = dir([input_resize6_png_Path,'/*.tif']);

%% DICE나 IoU를 계산할 파일 목록 불러오기 (필수)
% ROI 파일 이미지 목록 = DICE&IoU 계산할 샘플목록
clear dir;
allROIFiles = dir([input_ROI_Path,'/*.png']);

%% TIF를 RGB_dotting한 결과파일(*.mat) 목록 불러오기
% TIF를 RGB_dotting한 결과 폴더
allMatFiles = dir([input_mat_Path, '/*.mat']);



% tissuenames : 모델의 분류 카테고리
tissuenames = unique(myNet.Layers(end).ClassNames);

% 병변(tumor) 클래스 one-hot 인코딩으로 변경
% C-04 = tumor 버전
TUM_class = ["C-01" "C-02" "C-03" "C-04" "C-05"];
% 기본값 false인 15x1 배열을 생성
TUM_idx = false(numel(tissuenames), 1);
% tissuenames 개수만큼 순서대로 TUM_class와 일치하는 인덱스에 true값을 세팅
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == TUM_class) > 0
        TUM_idx(var_i, 1) = true;
    end
end
% TUM_class와 개수가 같은지 확인
sum(TUM_idx)

% Others 클래스
others_class = ["C-13" "C-14"];
others_idx = false(numel(tissuenames), 1);
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == others_class) > 0
        others_idx(var_i, 1) = true;
    end
end
sum(others_idx)

% 병변 및 비병변 분율을 계산할때 제외할
% 기타 클래스 설정 (EDGE, RBC, BACK)
out_class = ["C-15"];
out_idx = false(numel(tissuenames), 1);
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == out_class) > 0
        out_idx(var_i, 1) = true;
    end
end
sum(out_idx)

% one-hot 인코딩으로 변경
Normal_idx = ~TUM_idx;
Normal_idx = Normal_idx & ~others_idx;
Normal_idx = Normal_idx & ~out_idx;
sum(Normal_idx)

% 모두 합쳐서 tissuenames와 개수가 같아야 함
disp(strcat("모델 분류 클래스 개수 : ", num2str(numel(tissuenames))));
disp(strcat("one-hot 인코딩 전체 클래스 개수 : ", num2str(sum(TUM_idx|Normal_idx|others_idx|out_idx))));

% C-04를 tumor에 있으면 tumor(red)로 아니면 normal(green)으로 변경
% C-04 = tumor 버전
if sum(TUM_class == 'C-04') == 1
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14          C-15     };
%     colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
%     colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;];
    colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;];
else % C-04 = normal 버전
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14         C-15     };
%     colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
%     colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;];
    colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;];
end

% AI가 분류하는 클래스의 차원 인덱스
class_num = 1:numel(tissuenames);

filter_size_i = 1;

for filter_size_i=1:numel(arr_size_filter)
    % sigma 별로 엑셀파일을 출력하기위해 초기화
    clearvars allSummaryROIFiles;

    % 필터를 세팅
    nFilterSize = arr_size_filter(filter_size_i);

    file_i = 1;
    for file_i = 1:numel(all_sample_list_string)

        % 1. sample 이름을 추출한다.
        sample_name = strrep(all_sample_list_string(file_i), '-', '_');
        disp(strcat("filei : ", num2str(file_i), "  /  sample name : ", sample_name));


        % 2. ROI 파일 목록에서 같은 샘플이름을 가진 ROI 파일 유무 체크
        b_ExistROI = false;
        % ROI_idx = 1;
        for ROI_idx = 1:numel(allROIFiles)
            ROI_name = strrep(allROIFiles(ROI_idx).name, '-', '_');

            ROI_path = [allROIFiles(ROI_idx).folder,'/',allROIFiles(ROI_idx).name];

            is_equal_sample = strfind(upper(ROI_name), upper(sample_name));
            if numel(is_equal_sample) == 1
                b_ExistROI = true;
                break
            end
        end

        if b_ExistROI
            disp(strcat("ROI 파일을 찾았습니다 ROI_name : ", ROI_name));
        else
            disp(strcat("엑셀의 ", num2str(file_i), "번째 sample : ", sample_name, "과 동일한 ROI 파일을 찾지 못했습니다!!!!! ROI 파일 여부를 확인하세요"));
            return;
        end
        % 2. ROI 파일 목록에서 같은 샘플이름을 가진 ROI 파일 찾기 종료

        % 3. mat 파일 목록에서 같은 샘플이름을 가진 분석결과(mat) 파일 유무 체크
        b_ExistMat = false;
        % mat_idx = 1;
        for mat_idx = 1:numel(allMatFiles)
            mat_name = strrep(allMatFiles(mat_idx).name, '-', '_');

            mat_path = [allMatFiles(mat_idx).folder,'/',allMatFiles(mat_idx).name];
            % sample_name이 포함된 mat파일을 찾는다.
            is_equal_sample = strfind(upper(mat_name), upper(sample_name));
            if numel(is_equal_sample) == 1
                b_ExistMat = true;
                break
            end
        end
        if b_ExistMat
            disp(strcat("mat 파일을 찾았습니다 mat_name : ", mat_name));
        else
            disp(strcat("엑셀의 ", num2str(file_i), "번째 sample : ", sample_name, "과 동일한 mat 파일을 찾지 못했습니다!!!!! mat 파일 여부를 확인하세요"));
            return;
        end

        % structure 변수 newData1에 mat파일안에 저장한 변수들을 불러온다
        newData1 = load('-mat', mat_path);
        % newData1.mask 변수값을 base 작업공간에 mask 변수에 넣는다
        assignin('base', 'mask', newData1.('mask'));

        % mat파일 만들때 사용한 border
        % 기본 [12 12] mat파일에서 불러와서 덮어씌움
        border = [12 12];

        % newData1.mask 변수값을 base 작업공간에 mask 변수에 넣는다
        try % 이전에 저장한 mat파일에 border가 없어 error가 나는것을 대비
            assignin('base', 'border', newData1.('border'));
        catch
        end
        disp(strcat("border : ", num2str(border(1)), ", ", num2str(border(2))))

        % newData1.model_Rev 변수 값을 base 작업공간에 mask 변수로 넣는다
        assignin('base', 'model_Rev_by_mat', newData1.('model_Rev'));

        % mat파일과 AI 모델 Rev가 동일한지 체크하여 안맞으면 에러메시지를 뱉고 다음 ROI로 넘어간다.
        %         if string(model_Rev_by_mat) ~= string(model_Rev)
        %             disp("mat파일을 만든 AI 버전이 AI 모델과 일치하지 않습니다!!!!!!!");
        %             disp(strcat("분석용 모델파일의 Rev :", model_Rev));
        %             disp(strcat('mat 파일의 model Rev : ', model_Rev_by_mat'));
        %             return;
        %         end

        bsize = bsize(1:2)-2*border;
        % 패딩에 영향받은 가장자리 영역 설정
        rmov = ceil(border(1)/bsize(1));

        %% RGB dotting을 위한 세팅
        if rmov > 0
            mask_margin = mask((rmov):(end-rmov),(rmov):(end-rmov),:);
        else
            % rmov가 0인경우 인덱스 오류가 나서 처리
            mask_margin = mask((1):(end),(1):(end),:);
        end
        % mask_margin의 확률값을 어떻게 이미지로 변환하는가에 대한 원리 설명
        % mask_margin(1, 1, 1:15)
        %
        % for i = 1:164
        %     for j = 1:266
        %         for k=1:15
        %             a(k) = mask_margin(i, j, k)
        %             max_a = max(a);
        %             % a의 최대값을 꺼내올 수 있음
        %             a(a == max_a) % 0 0 0 0 0 0 0 ... 1
        %             % colors_TN의 15번째 값을 꺼내온다
        %                colors_TN(15, :)
        %                colors_TN(a == max_a)
        %     end
        % end
        
        mask_size = size(mask_margin);

        % 같은 병리번호로 H&E mini 사이즈 파일 목록에서 ROI_name과 동일한 이름 유무 체크
        b_ExistPNG = false;
        % PNG_idx = 1;
        for PNG_idx = 1:numel(allPNGFiles)
            PNG_name = strrep(allPNGFiles(PNG_idx).name, '-', '_');

            PNG_path = [allPNGFiles(PNG_idx).folder,'/',allPNGFiles(PNG_idx).name];

            is_equal_sample = strfind(upper(PNG_name), upper(sample_name));
            if numel(is_equal_sample) == 1
                b_ExistPNG = true;
                break
            end
        end

        if b_ExistPNG
            disp(strcat("H&E 파일을 찾았습니다 PNG_name : ", PNG_name));
            % H&E 파일을 불러온다.
            [curImage_org, map, alpha] = imread(PNG_path);
            %       imshow(curImage_org) % H&E 이미지를 Plot창에 출력
            % curImage_resize : H&E 이미지를 ROI 사이즈에 맞춰 리사이징
            curImage_resize = imresize(curImage_org, mask_size(1:2));
        end

        % ROI 파일을 불러온다.
        [curImage_ROI_org, map, alpha] = imread(ROI_path);
        %       imshow(curImage_ROI_org) % ROI 이미지를 Plot창에 출력

        curImage_ROI_org_size = size(curImage_ROI_org);
        
        % 2022-08-23 from. 장은지
        % ROI 색상을 흑/백(Tumor)로 변경하여 리사이징
        curImage_ROI_org_gray = im2gray(curImage_ROI_org);
        curImage_ROI_org_binary = imbinarize(curImage_ROI_org_gray, 'global');

% figure
% nexttile; imshow(curImage_ROI_org)
% nexttile; imshow(curImage_ROI_org_gray)
% nexttile; imshow(curImage_ROI_org_binary)
        %         curImage_ROI_org_binary_resize = imresize(curImage_ROI_org, mask_size(1:2));
        % "nearest" "bilinear" "bicubic"
        % curImage_ROI_org_binary_resize : 천공구역 색칠한 이미지를 mask 사이즈에 맞게 리사이징
        curImage_ROI_org_binary_resize = imresize(curImage_ROI_org_binary, mask_size(1:2), "bicubic");
% nexttile; imshow(curImage_ROI_org_binary_resize)
        % activation 값을 이용하여 max인 클래스의 RGB 색상으로 전환
        
        %% 샘플 영역 예측확률값중 가장 큰 확률값을 가진 클래스의 RGB 색상을 가져오고 해당 클래스의 개수를 카운트
        % max_class_idx : activation의 최대값을 찾아 최대값의 차원 인덱스를 가진다
        max_class_idx = zeros(mask_size(1), mask_size(2));

        % rgbout : activation의 최대값인 클래스의 인덱스를 찾아 colors(RGB)의 해당 인덱스값으로 변환
        rgbout = zeros(mask_size(1), mask_size(2), 3, 'uint8');

        % Tumor class or Normal class 여부(1 or 0) 행렬을 생성하여
        % ROI와 얼마나 겹치는지 여부를 측정해보자
        colors_class_TN = zeros(mask_size(1), mask_size(2), 'uint8');

        % rgbout_TN : Red(tumor) vs Green색(normal) RGB 값을 가진 m x매트릭스
        rgbout_TN = zeros(mask_size(1), mask_size(2), 3, 'uint8');

        % count_class_by_ALL : 전체 샘플영역 클래스별 pixel 개수
        count_class_by_ALL = zeros(1, numel(tissuenames));

        % count_class_by_ROI : ROI 영역안에 클래스별 pixel 개수
        count_class_by_ROI = zeros(cutoff_color_cnt, numel(tissuenames));

        % imfuse 시 표시할 새하얀 (1 1 1) RGB 매트릭스를 생성해본다
        size_curImage_ROI_resize = size(curImage_ROI_org_binary_resize);
        rgb_imfuse_roi = ones(size_curImage_ROI_resize(1), size_curImage_ROI_resize(2), 3);

        j=1;
        k=1;
        m=1;

        % 2022-03-02 : 확률 매트릭스를 가우스필터링 한 후
        for j=1:mask_size(1)
            for k=1:mask_size(2)
                % 1. max(mask_margin(j,k,:)) : mask_margin(j, k)번째 15개 클래스중 가장 큰 확률값
                % 2. (mask_margin(j,k,:) == 1에서구한 가장큰 확률값) : 확률값의 인덱스 logical 값
                % 3. class_num(가장 큰 확률값의 인덱스 logical 값)
                % 가장 큰 확률값의 인덱스 logical 값 예시
                % temp_class_num = class_num([0 0 1 0 0 0 0 0 0 0 0 0 0 0 0])
                % temp_class_num = 3;
                temp_class_num = class_num(mask_margin(j,k,:)==max(mask_margin(j,k,:)));
                % 간혹 동일한 확률값이 2개 이상인 경우 첫번째 확률이 같은 클래스를 세팅
                % 더 중요한 클래스인 병변 클래스가 클래스 앞쪽에 위치하므로
                max_class_idx(j, k) = temp_class_num(1);

                rgbout(j, k, 1:3) = colors(max_class_idx(j,k),1:3);

                rgbout_TN(j, k, 1:3) = colors_TN(max_class_idx(j,k),1:3);
                
                % out_idx에 속한 클래스는 카운트에서 제외한다.
                % 2023-02-01 / 김현태 : 검산에 용이하도록 카운트하게 변경한다.
%                 if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
%                     continue
%                 end

                % if문으로
                %max_class_TN(j, k) = sum((reshape(TUM_idx, [1 numel(class_num)]) & (temp_class_num(1) == class_num))) == 1;
                colors_class_TN(j, k) = colors_TN_gray(max_class_idx(j,k));

                % 전체 샘플의 클래스별 pixel 개수 카운트
                count_class_by_ALL(1, max_class_idx(j, k)) = count_class_by_ALL(1, max_class_idx(j, k)) + 1;

                % ROI영역의 색상 개수별로 반복
                % ROI영역의 색상 개수별로 반복
                for color_i = 1:cutoff_color_cnt
                    %                 if sum(curImage_ROI_org_binary_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(color_i, :), [1 1 3])) == 3
                    % 2022-08-23
                    if (curImage_ROI_org_binary_resize(j, k, :) == 0)
                        % RGB 채널의 값이 모두 같으면(같은색상) 해당 ROI 영역색상별 클래스별 pixel 개수 카운트
                        count_class_by_ROI(color_i, max_class_idx(j, k)) = count_class_by_ROI(color_i, max_class_idx(j, k)) + 1;
                        rgb_imfuse_roi(j, k, :) = [1 0 1];
                    end
                end
            end
        end
        
% 중간 검산 그리드 크기별 클래스 카운팅
% disp("RGB 분석한 전체 pixel 카운트 개수")
% disp(size_curImage_ROI_resize(1) * size_curImage_ROI_resize(2))
% disp("클래스별 카운트 개수")
% disp(count_class_by_ALL)
% disp("전체합(클래스별 카운트 개수)")
% disp(sum(count_class_by_ALL))
% disp("ROI 영역의 전체 클래스별 카운트 개수")
% disp(count_class_by_ROI)
% disp(sum(count_class_by_ROI))

        %% 이미지 Plot 출력
        if bShowPlot
            figure("Position", [0 0 1024 768]);
            tiledlayout('flow');
            %             whitebg([0 .5 .6])
            % 2022-08-19 / from. 장은지 : 최종출력 이미지 조정
            % 1. HE & ROI imfuse
            img_imfuse = imfuse(rgb_imfuse_roi, curImage_resize, 'blend');
            nexttile; imshow(img_imfuse); title("WSI with ROI");
        end

        %% 블러처리 전 분석상태에서 DICE 계수를 구해본다.
        % tot_sample_ALL : 전체 영역 샘플 pixel 개수 합계를 구한다.
        tot_sample_ALL = sum(count_class_by_ALL(1, TUM_idx|Normal_idx|others_idx|out_idx));

        % tot_tumor_ALL : 전체 영역 Tumor 클래스의 pixel 개수 합계를 구한다.
        tot_tumor_ALL = sum(count_class_by_ALL(1, TUM_idx));

        % tot_normal_ALL : 전체 영역 Normal 클래스의 pixel 개수 합계를 구한다.
        tot_normal_ALL = sum(count_class_by_ALL(1, Normal_idx));

        % tot_others_ALL : 전체 영역 Other 클래스의 pixel 개수 합계를 구한다.
        tot_other_ALL = sum(count_class_by_ALL(1, others_idx));
        
        % tot_out_ALL : 전체 영역 Background 클래스의 pixel 개수 합계를 구한다.
        tot_out_ALL = sum(count_class_by_ALL(1, out_idx));

        % 카운트 검산 test
        %       disp(tot_sample_ALL)
        %       disp(sum([tot_tumor_ALL, tot_normal_ALL, tot_other_ALL, tot_out_ALL]))

        %% cutoff 세팅별로 Blur 필터를 적용
        max_i=1;
        for max_i=1:numel(arr_cutoff_max)
            disp(strcat('max_i : ', num2str(max_i), " / ", num2str(numel(arr_cutoff_max))));

            % allSummaryROIFiles 구조체 변수에 출력할 정보 세팅
            % slidename : 샘플 정보
            % mat_name  :
            % color : ROI 색상정보
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).slide_name = sample_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).mat_name = mat_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).color = cutoff_color_name(max_i);

            % 이미지 필터를 해서 노이즈 제거 실시
            if (nFilterSize ~= 0)
                % 이미지 필터
                if str_filter_func == "imfilter"
                    nSigma_filter = ones(nFilterSize, nFilterSize);
                    nSigma_filter = nSigma_filter / (nFilterSize*nFilterSize);
                    rgbout_TN_filtered = imfilter(rgbout_TN, nSigma_filter, 'replicate','same');
                    colors_class_TN_filtered = imfilter(colors_class_TN, nSigma_filter, 'replicate', 'same');
                elseif str_filter_func == "imgaussfilt"
                    rgbout_TN_filtered = imgaussfilt(rgbout_TN, nFilterSize);
                    colors_class_TN_filtered = imgaussfilt(colors_class_TN, nFilterSize);
                end
            else
                rgbout_TN_filtered = rgbout_TN;
                colors_class_TN_filtered = colors_class_TN;
            end

            % % 중간 이미지 확인용 테스트 스크립트
            if bShowPlot
                nexttile; imshow(rgbout_TN);                title("AI Analyze 3 colors");
                nexttile; imshow(colors_class_TN);          title("AI Analyze grayscale");
                
                nexttile; imshow(rgbout_TN_filtered);       title(strcat(str_filter_func, " AI Analyze 3 colors"));
                nexttile; imshow(colors_class_TN_filtered); title(strcat(str_filter_func, " AI Analyze grayscale"));
                
            end
            % cutoff_max : arr_green_max에 max_i번째 세팅값
            cutoff_max = arr_cutoff_max(max_i);
            %% 이미지 필터 후 필터 크기별 반복
            % count_class_TN_by_ALL : 블러 전체 샘플영역 클래스 카운트
            count_class_TN_by_ALL = zeros(1, numel(tissuenames_category));
            % count_class_TN_by_ROI : curImage_ROI_resize에서 색칠한 색상 영역별 클래스 개수
            count_class_TN_by_ROI = zeros(1, numel(tissuenames_category));
            % count_class_TN_by_REST : curImage_ROI_resize에서 색칠한 색상 영역 바깥 클래스 개수
            count_class_TN_by_REST = zeros(1, numel(tissuenames_category));

            % ** 필터별 * Green 색상 쓰레스홀드별 샘플 클래스 pixel 개수 카운트
            j = 1;
            k = 1;

            % 카운팅할 최종 카테고리 숫자 인덱스를 세팅
            % tissuenames_category의 개수와 관련있음
            n_category = 0;
            
            for j=1:mask_size(1)
                for k=1:mask_size(2)
                    % 2020-09-09 : 여기서 픽셀별로 jpg 색상이랑 비교해서 카운팅 해보자
                    % 제외할 클래스는 색상 컨트롤에서 제외함
                    if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                        %                         disp(strcat(num2str(j), "/", num2str(k), ":", num2str(max_class_idx(j, k))))
                        %                         continue % 검산을 위해 BACK도 카운트 해보자
                        % Back 카테고리 카운트
                        n_category = 4;

                        % other (회색)으로 이진화
                        set_color_gray = colors_TN_gray(out_idx, :);
                        set_color_gray = set_color_gray(1); % 첫번째 색상 사용

                        % RGB_TN의 색상은 other(회색) 색상으로 이진화한다
                        set_color_RGB = colors_TN(out_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % 첫번째 색상 사용

                    elseif sum(max_class_idx(j, k) == class_num(others_idx)) > 0
                        % other 카테고리 카운트
                        n_category = 3;
                        % other (회색)으로 이진화
                        set_color_gray = colors_TN_gray(others_idx, :);
                        set_color_gray = set_color_gray(1); % 첫번째 색상 사용

                        % RGB_TN의 색상은 other(회색) 색상으로 이진화한다
                        set_color_RGB = colors_TN(others_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % 첫번째 색상 사용

                    elseif colors_class_TN_filtered(j, k) >= cutoff_max
                        % 블러링한 흑백 이미지의 명암값이 cutoff 값보다 값이 같거나 크다면
                        % 카운팅할 최종 카테고리는 Tumor
                        % tissuenames_category의 2번째가 Tumor
                        n_category = 1;

                        % Tumor(흰색)으로 이진화
                        set_color_gray = colors_TN_gray(TUM_idx, :);
                        set_color_gray = set_color_gray(1); % 첫번째 색상 사용

                        % RGB_TN의 색상은 normal(그린) 색상으로 이진화한다
                        set_color_RGB = colors_TN(TUM_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % 첫번째 색상 사용

                    elseif colors_class_TN_filtered(j, k) < cutoff_max
                        % 블러링한 흑백 이미지의 명암값이 cutoff 값보다 값이 작다면
                        % 카운팅할 최종 카테고리는 Normal
                        % tissuenames_category의 2번째가 Normal
                        n_category = 2;

                        % normal(검정색)으로 이진화
                        set_color_gray = colors_TN_gray(Normal_idx, :);
                        set_color_gray = set_color_gray(1); % 첫번째 색상 사용

                        % RGB_TN의 색상은 normal(그린) 색상으로 이진화한다
                        set_color_RGB = colors_TN(Normal_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % 첫번째 색상 사용
                    end

                    % 블러처리한 colors_class_TN_filtered 이미지를 이진화한다.
                    colors_class_TN_filtered(j, k) = set_color_gray;

                    % 블러처리한 rgbout_TN_filtered 이미지를 이진화한다.
                    rgbout_TN_filtered(j, k, :) = set_color_RGB;

                    % 전체영역의 Normal 카운트 + 1
                    count_class_TN_by_ALL(1, n_category) = count_class_TN_by_ALL(1, n_category) + 1;
                    % ROI 영역에 속한 픽셀인지 검사한다
                    % ROI 영역 안쪽이면 count_class_TN_by_ROI에 카운트 + 1
                    % ROI 영역 바깥쪽이면 count_class_TN_by_REST에 카운트 + 1
                    %                     if sum(curImage_ROI_org_binary_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                    if (curImage_ROI_org_binary_resize(j, k, :) == 0)
                        count_class_TN_by_ROI(1, n_category) = count_class_TN_by_ROI(1, n_category) + 1;
                    else
                        count_class_TN_by_REST(1, n_category) = count_class_TN_by_REST(1, n_category) + 1;
                    end
                end
            end

            % 중간점검 : 필터로 뭉개진 부위를 카테고리별로 나누어 후처리한 이미지 표시
            if bShowPlot
                nexttile; imshow(rgbout_TN_filtered);       title(strcat("post processing", " AI Analyze 3colors"));
                nexttile; imshow(colors_class_TN_filtered); title(strcat("post processing", " AI Analyze grayscale"));
            end

            % 전체 영역 픽셀 카운트 변수
            % cnt_all_sum : 샘플의 전체 클래스 개수 합계를 구한다.
            cnt_all_sum = sum(count_class_TN_by_ALL(1, :));
            % cnt_all_tumor : 샘플의 Tumor 클래스의 합계를 구한다.
            cnt_all_tumor = sum(count_class_TN_by_ALL(1, 1));
            % cnt_all_normal : 샘플의 Normal 클래스의 합계를 구한다.
            cnt_all_normal = sum(count_class_TN_by_ALL(1, 2));
            % cnt_all_others : 샘플의 Normal 클래스의 합계를 구한다.
            cnt_all_others = sum(count_class_TN_by_ALL(1, 3));
            % cnt_all_back : 샘플의 Normal 클래스의 합계를 구한다.
            cnt_all_back = sum(count_class_TN_by_ALL(1, 4));


            % ROI 영역 픽셀 카운트 변수
            %             % cnt_ROI_sum : ROI 영역 Tumor와 Normal과 Other, Backgroud의 전체 픽셀 개수를 구한다.
            cnt_ROI_sum = sum(count_class_TN_by_ROI(1, :));

% 검산용
disp( cnt_ROI_sum == sum(sum(curImage_ROI_org_binary_resize==false)) )

            % cnt_ROI_tumor : ROI 영역안에 Tumor 클래스의 합계를 구한다.
            cnt_ROI_tumor = sum(count_class_TN_by_ROI(1, 1));
            % cnt_ROI_normal : ROI 영역안에 Normal 클래스의 합계를 구한다.
            cnt_ROI_normal = sum(count_class_TN_by_ROI(1, 2));
            % cnt_ROI_others : 샘플의 Others 클래스의 합계를 구한다.
            cnt_ROI_others = sum(count_class_TN_by_ROI(1, 3));
            % cnt_ROI_back : 샘플의 Back 클래스의 합계를 구한다.
            cnt_ROI_back = sum(count_class_TN_by_ROI(1, 4));

% 검산용
% disp(strcat("ROI영역 전체 픽셀 개수 cnt_ROI_sum : ", num2str(cnt_ROI_sum)))
% disp(strcat("ROI영역 tumor 픽셀 개수 합 C-01~C05 : ", num2str(sum(count_class_TN_by_ROI(1, 1:4)))))
% disp(strcat("ROI영역 tumor 픽셀 개수 cnt_ROI_tumor : ", num2str(cnt_ROI_tumor)))
% disp(strcat("ROI영역 tumor 픽셀 개수 합 C-01~C05 : ", num2str(sum(count_class_TN_by_ROI(1, 1)))))
% disp(strcat("ROI영역 normal 픽셀 개수 cnt_ROI_normal : ", num2str(cnt_ROI_normal)))
% disp(strcat("ROI영역 normal 픽셀 개수 합 C-06~C12 : ", num2str(sum(count_class_TN_by_ROI(1, 2)))))
% disp(strcat("ROI영역 others 픽셀 개수 cnt_ROI_others : ", num2str(cnt_ROI_others)))
% disp(strcat("ROI영역 others 픽셀 개수 합 C-13~C14 : ", num2str(sum(count_class_TN_by_ROI(1, 3)))))
% disp(strcat("ROI영역 back 픽셀 개수 cnt_ROI_back : ", num2str(cnt_ROI_back)))
% disp(strcat("ROI영역 back 픽셀 개수 합 C-15 : ",     num2str(sum(count_class_TN_by_ROI(1, 4)))))

            % REST 영역 픽셀 카운트 변수
            % ROI 색상영역 바깥에 전체(other와 background 제외됨) 합계를 구한다.
            cnt_REST_sum = sum(count_class_TN_by_REST(1, :));

% 검산용
% disp( cnt_REST_sum == sum(sum(curImage_ROI_org_binary_resize==true)) )

            % cnt_REST_tumor : ROI 영역 바깥의 Tumor 합계를 구한다.
            cnt_REST_tumor = sum(count_class_TN_by_REST(1, 1));
            % cnt_REST_normal : ROI 영역 바깥의 Normal 합계를 구한다.
            cnt_REST_normal = sum(count_class_TN_by_REST(1, 2));
            % cnt_REST_others : ROI 영역 바깥의 Others 합계를 구한다.
            cnt_REST_others = sum(count_class_TN_by_REST(1, 3));
            % cnt_REST_back : ROI 영역 바깥의 Background 합계를 구한다.
            cnt_REST_back = sum(count_class_TN_by_REST(1, 4));
            

% 검산 Test
% disp(string(cnt_all_sum == cnt_ROI_sum + cnt_REST_sum))

            %% A. blur 처리 후 *ROI영역 안* + *Tumor* 계수를 구하기 위한 수식 검증
%             TP = cnt_ROI_tumor;                   % ROI 색상영역 안쪽의(Positive) Tumor(Positive) 클래스의 합계를 구한다. 
%             FP = cnt_REST_tumor;                  % ROI 색상영역 바깥의(Negative) Tumor(Positive) 합계를 구한다.
%             FN = cnt_ROI_normal+cnt_ROI_others;   % ROI 색상영역 안쪽의(Positive) Normal(Negative) 클래스의 합계를 구한다.
%             TN = cnt_REST_normal+cnt_REST_others; % ROI 색상영역 바깥의(Negative) Normal(Negative) 합계를 구한다.

            %% A-1. *ROI영역 안* DICE 계수 검산을 위한 수식
%             similarity = 2 * TP               / ( 2 * TP + FP + FN)
            % similarity = 2 * jaccard(A,B)     / (1 + jaccard(A,B) )
%                               2 *TP           / ((TP  + FN)              + (TP   + FP))
%                                                 (4714 + (287 + 360))     + (4714 + 1644)
%                              (2 * 4714)       / (5361                    + 6358)
%             similarity = (2 * cnt_ROI_tumor)  / (cnt_ROI_sum      + cnt_all_tumor)
% 
%                               2 * TP          / ( 2 * TP                + FP             + FN)
%                              (2 * 4714)       / ( 2 * 4714              + 1644           + (287 + 360) )
%             similarity = (2 * cnt_ROI_tumor)  / ((2 * cnt_ROI_tumor) + cnt_REST_tumor + (cnt_ROI_normal + cnt_ROI_others))
            % AI가 tumor로 예측한 영역 true
            AI_BW_tumor = colors_class_TN_filtered == 255;
            % 병리사의 tumor로 예측한 영역 true
            Doctor_BW_tumor = ~curImage_ROI_org_binary_resize;
            % MATLAB 제공 dice() 함수로 DICE 계산
            DICE_ROI_tumor = dice(AI_BW_tumor, Doctor_BW_tumor);

            % dice ROI Tumor 계산 점검
            if bShowPlot
                nexttile; imshow(AI_BW_tumor);                         title(strcat('AI_BW_tumor'), 'Interpreter', 'none');
                nexttile; imshow(Doctor_BW_tumor);                   title(strcat('Doctor_BW_tumor'), 'Interpreter', 'none');
                nexttile; imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('DICE_ROI_tumor : ', num2str(round(DICE_ROI_tumor, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['DICE_ROI_tumor', num2str(nFilterSize)]) = DICE_ROI_tumor;
            

            %% A-2. *ROI영역 안* IoU 검산을 위한 수식
%             % 기존방식->       TP                /    ((TP + FN)            + (TP + FP)              - TP)
%                             4714                 /    ((4714 + (287 + 360)) + (4714+1644)            - 4714)
%                             4714                 /    (5361                 + 6358                   - 4714)
%                       cnt_ROI_tumor              / (cnt_ROI_sum       + cnt_all_tumor    - cnt_ROI_tumor)
%             % blur 처리 후 ROI 색상영역 안 Tumor IoU 계수
%             IoU_ROI_tumor = (cnt_ROI_tumor) / (cnt_ROI_sum + cnt_all_tumor-cnt_ROI_tumor);            

%           % 변경한 방식->    TP                  /     (TP                  + FP                      + FN)
%                             4714                 /     (4714                + 1644                    + (287 + 360))
%                         cnt_ROI_tumor            / (cnt_ROI_tumor + cnt_REST_tumor + cnt_ROI_normal + cnt_ROI_others)
%             IoU_ROI_tumor = cnt_ROI_tumor / (cnt_ROI_tumor + cnt_REST_tumor + cnt_ROI_normal + cnt_ROI_others)
            
            % MATLAB 제공 jaccard() 함수로 IoU 계산
            IoU_ROI_tumor = jaccard(AI_BW_tumor, Doctor_BW_tumor);

            % IoU ROI Tumor 계산 점검
            if bShowPlot
                nexttile; imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('IoU_ROI_tumor : ', num2str(round(IoU_ROI_tumor, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['IoU_ROI_tumor', num2str(nFilterSize)]) = IoU_ROI_tumor;
            
            %% B. blur 처리 후 *(가상의 normal ROI = tumor ROI의 반대)* + *Normal* 계수를 구하기 위한 수식 검증
% TN = cnt_ROI_tumor+cnt_ROI_others;   % (TP->FN) ROI 색상영역 안쪽의(Positive) Tumor(Negative) 클래스의 합계를 구한다. 
% FN = cnt_REST_tumor+cnt_REST_others; % (FP->TN) ROI 색상영역 바깥의(Negative) Tumor(Negative) 합계를 구한다.
% FP = cnt_ROI_normal;                 % (FN->FP) ROI 색상영역 안쪽의(Positive) Normal(Positive) 클래스의 합계를 구한다.
% TP = cnt_REST_normal;                % (TN->TP) ROI 색상영역 바깥의(Negative) Normal(Positive) 합계를 구한다.
            %% B-1. *(가상의 normal ROI = tumor ROI의 반대)* DICE 계수 검산을 위한 수식
%             similarity = 2 * TP                / ( 2 * TP + FP + FN)
%                               2 * TP           / ( 2 * TP                + FP             + FN)
%             similarity = (2 * cnt_REST_normal) / ((2 * cnt_REST_normal) + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others))

            % AI가 normal로 예측한 영역 (true)
            AI_BW_normal = colors_class_TN_filtered == 0;
            
            % AI가 background 로 예측한 영역 (true)
            AI_BW_back = (colors_class_TN_filtered == 128);
            % 병리사의 normal로 예측한 영역을 가상으로 생성
            % 기존 ROI를 반전한다음 AI가 back그라운드로 예측한 영역을 false처리
            % 배경 제거(tumor = false, normal = true, back = false)            
            Doctor_BW_normal = xor(curImage_ROI_org_binary_resize, AI_BW_back);
%             Doctor_BW_normal = curImage_ROI_org_binary_resize == ~AI_BW_back;
            DICE_ROI_normal = dice(AI_BW_normal, Doctor_BW_normal);

% dice 계산 점검
% IoU ROI Tumor 계산 점검
            if bShowPlot
                nexttile; imshow(AI_BW_normal);                           title(strcat('AI_BW_normal'), 'Interpreter', 'none');
                nexttile; imshow(Doctor_BW_normal);                       title(strcat('Doctor_BW_normal(if)'), 'Interpreter', 'none');
                nexttile; imshow(imfuse(AI_BW_normal, Doctor_BW_normal)); title(strcat('DICE_ROI_normal : ', num2str(round(DICE_ROI_normal, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['DICE_ROI_normal', num2str(nFilterSize)]) = DICE_ROI_normal;
            
            %% B-2. *(가상의 normal ROI = tumor ROI의 반대)* IoU 계수 검산을 위한 수식
            % blur 처리 후 ROI 색상영역 Normal IoU 계수
            % similarity =    TP                /  (TP                      + FP                       + FN)
            %             cnt_REST_normal / (cnt_REST_normal + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others))            
%             IoU_ROI_normal = cnt_REST_normal / (cnt_REST_normal + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others));
            % MATLAB 제공 jaccard() 함수로 IoU 계산
            IoU_ROI_normal = jaccard(AI_BW_normal, Doctor_BW_normal);
            % IoU ROI Tumor 계산 점검
            if bShowPlot
                nexttile; imshow(imfuse(AI_BW_normal, Doctor_BW_normal)); title(strcat('IoU_ROI_normal : ', num2str(round(IoU_ROI_normal, 3))), 'Interpreter', 'none');
            end
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['IoU_ROI_normal', num2str(nFilterSize)]) = IoU_ROI_normal;

            % allSummaryROIFiles : 최종출력 엑셀파일에 카운팅한 픽셀개수를 저장하기 위함
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_SUM    = cnt_all_sum;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_tumor  = cnt_all_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_normal = cnt_all_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_others = cnt_all_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_back   = cnt_all_back;
            
            % 검산용 항목 추가
            % tot_tumor_normal_SUM 항목과 tot_ROI_REST_SUM 항목은 숫자가 같아야 한다
%             allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).tot_tumor_normal_SUM = cnt_all_tumor + cnt_all_normal;
%             allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).tot_ROI_REST_SUM = cnt_ROI_tumor + cnt_ROI_normal + cnt_REST_tumor + cnt_REST_normal;
            % 검산용 항목 추가

            % ROI 영역
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_tumor   = cnt_ROI_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_normal  = cnt_ROI_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_others  = cnt_ROI_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_back    = cnt_ROI_back;
            
            % ROI 바깥 영역
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_tumor  = cnt_REST_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_normal = cnt_REST_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_others = cnt_REST_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_back   = cnt_REST_back;

            if bShowPlot
                print(gcf,'-djpeg','-r300', strcat(output_path,'/', sample_name, '_filter_size_', num2str(nFilterSize), '_scen_', num2str(cutoff_max*100), '_', str_filter_func, '.jpg'));
            end
        end
        close all;
    end
    % 필터 크기별로 엑셀파일을 출력한다.
    strToday = datestr(now,'yyyymmdd_HHMM');
    save_allSummaryROIFiles = struct2table(allSummaryROIFiles);
    writetable(save_allSummaryROIFiles, strcat(output_path, '/', model_Rev, '_', num2str(numel(allROIFiles)), '개샘플_ROI별_클래스분류_카운트_filter_size_', num2str(nFilterSize), '_', str_filter_func, '_', strToday,'.xlsx'));
end