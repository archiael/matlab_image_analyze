%% 2022-08-18 DICE와 IoU 계수 계산 스크립트 개발
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

bShowPlot = true;

gpuDevice(2);

%% 분석에 필요한 변수 세팅

% 분석할 샘플 목록 엑셀파일을 읽어온다.
sample_list = readtable("E:\git\AI_model_kimhyuntae\8.analysis_by_ROI\analysis_for_DICE_IoU_by_ROI.xlsx");

% 엑셀 헤더 정보 출력
disp(sample_list.Properties.VariableNames(:));
disp(numel(sample_list.Properties.VariableNames));

% cell을 string 형으로 변환
sample_list_string = string(sample_list.filename(:));
disp(numel(sample_list_string));

% RGB에 천공 구획 표시 색상
% str_filter_func : imfilter(평균필터) / imgaussfilt(가우스 평활화 필터)
str_filter_func = "imfilter";
% arr_size_filter = [2]; % 가우시안 필터 조건식 => 2*ceil(2*sigma)+1
arr_size_filter = [9]; % 평균 필터 조건  9px일때 1mm에 근접

% 필터 후 cutoff 쓰레스홀드 수치별 비교용
% 최적 cutoff 수치는 0.5~0.6
arr_green_max = [0:0.1:1];

% 2022-07-19 ROI 색상은 이제 1가지만 되었으니 ROI 색상 변수를 반복할 green cutoff용으로 지정
% cutoff_color_name : 반복할 cutoff 개수를 지정
cutoff_color_name = string(arr_green_max);
arr_green_max_cnt = numel(arr_green_max);

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
input_png_Path = 'Z:\SET_003_LEICA\2.resize_6%';
allPNGFiles = dir([input_png_Path,'/*.png']);

%% DICE나 IoU를 계산할 파일 목록 불러오기 (필수)
% ROI 파일 이미지 목록 = DICE&IoU 계산할 샘플목록
clear dir;
% tuning 66 샘플
% input_ROI_Path = 'Z:\SET_003_LEICA\5.ROI\5.2.analysis\2.tuning\1.ROI_svg_to_png';

% test 99 샘플
input_ROI_Path = 'Z:\SET_003_LEICA\5.ROI\5.2.analysis\3.test\1.ROI_svg_to_png';

% TCGA surgical 187 샘플
% input_ROI_Path = 'Z:\SET_013_TCGA\TCGA.STAD\5.ROI\2.analysis\20220811';

allROIFiles = dir([input_ROI_Path,'/*.png']);

%% TIF를 RGB_dotting한 결과파일(*.mat) 목록 불러오기
% TIF를 RGB_dotting한 결과 폴더
% tuning
% inputPath = 'Z:\SET_003_LEICA\3.RGB_dotting\Rev0.01.13_LEICA_C04C05_tuning_66ea\mat';

% test
inputPath = 'Z:\SET_003_LEICA\3.RGB_dotting\Rev0.01.13_LEICA_C04C05_test_99ea\mat';

% TCGA
% inputPath = 'Z:\SET_013_TCGA\TCGA.STAD\3.RGB_dotting\Rev0.01.13_LEICA_C04C05\mat';
allMatFiles = dir([inputPath, '/*.mat']);

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
% myNet에 저장한 모델에서 Input 이미지 크기를 가져온다.
% VGG19모델은 기본 224x224x3 크기임
layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize;

clearvars testing_tbl training_categories training_tbl vars newData1;

%% DICE & IoU 계산한 결과를 저장할 폴더 지정
output_path = strcat(input_ROI_Path, '\DICE_IoU_계수_보고서_', model_Rev);
mkdir(output_path);
winopen(output_path);

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
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
    colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
else % C-04 = normal 버전
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14         C-15     };
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
    colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
end

% 병변, 비병변 2개로 나누어 카운트 진행
% Others를 
tissuenames_TN = {'Tumor', 'Normal', 'Others'};
class_num = 1:numel(tissuenames); % 분류 클래스의 차원 인덱스

sigma_i = 1;

for sigma_i=1:numel(arr_size_filter)
    % sigma 별로 엑셀파일을 출력하기위해 초기화
    clearvars allSummaryROIFiles;
    
    % 필터를 세팅
    nSigma = arr_size_filter(sigma_i);

    file_i = 1;
    for file_i = 170:numel(sample_list_string)
        
        % 1. sample 이름을 추출한다.
        sample_name = strrep(sample_list_string(file_i), '-', '_');
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
        
        if ~b_ExistROI
            disp(strcat("엑셀의 ", num2str(file_i), "번째 sample : ", sample_name, "과 동일한 ROI 파일을 찾지 못했습니다!!!!! ROI 파일 여부를 확인하세요"));
            return;
        else
            disp(strcat("ROI 파일을 찾았습니다 ROI_name : ", ROI_name));
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
        if ~b_ExistMat
            disp(strcat("엑셀의 ", num2str(file_i), "번째 sample : ", sample_name, "과 동일한 mat 파일을 찾지 못했습니다!!!!! mat 파일 여부를 확인하세요"));
            return;
        else
            disp(strcat("mat 파일을 찾았습니다 mat_name : ", mat_name));
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
            % H&E 파일을 불러온다.
            [curImage_org, map, alpha] = imread(PNG_path);
%       imshow(curImage_org) % H&E 이미지를 Plot창에 출력
        end
    
        % ROI 파일을 불러온다.
        [curImage_ROI_org, map, alpha] = imread(ROI_path);
%       imshow(curImage_ROI_org) % ROI 이미지를 Plot창에 출력

        curImage_ROI_org_size = size(curImage_ROI_org);
    
        %% RGB dotting을 위한 세팅
        if rmov > 0
            mask_margin = mask((rmov):(end-rmov),(rmov):(end-rmov),:);
        else
            % rmov가 0인경우 인덱스 오류가 나서 처리
            mask_margin = mask((1):(end),(1):(end),:);
        end

        mask_size = size(mask_margin);
        
        % curImage_ROI_resize : 천공구역 색칠한 이미지를 mask 사이즈에 맞게 리사이징
        curImage_ROI_resize = imresize(curImage_ROI_org, mask_size(1:2));
        
        % curImage_resize : H&E 이미지를 ROI 사이즈에 맞춰 리사이징
        if b_ExistPNG
            curImage_resize = imresize(curImage_org, curImage_ROI_org_size(1:2));
        end
        % activation 값을 이용하여 max인 클래스의 RGB 색상으로 전환
        
        %% 샘플 영역 예측값중 가장 큰 확률값을 가진 클래스의 RGB 색상을 계산하고 해당 클래스의 개수를 카운트
        % max_class_idx : activation의 최대값을 찾아 최대값의 차원 인덱스를 가진다
        max_class_idx = zeros(mask_size(1), mask_size(2));
        
        % rgbout : activation의 최대값인 클래스의 인덱스를 찾아 colors(RGB)의 해당 인덱스값으로 변환
        rgbout = zeros(mask_size(1), mask_size(2), 3);

        % Tumor class or Normal class 여부(1 or 0) 행렬을 생성하여 
        % ROI와 얼마나 겹치는지 여부를 측정해보자
        max_class_TN = zeros(mask_size(1), mask_size(2));
        
        % rgbout_TN : Red(tumor) vs Green색(normal) RGB 값을 가진 m x매트릭스
        rgbout_TN = zeros(mask_size(1), mask_size(2), 3); 
        
        % count_class_by_ALL : 전체 샘플영역 클래스별 pixel 개수
        count_class_by_ALL = zeros(1, numel(tissuenames));

        % count_class_by_ROI : ROI 영역안에 클래스별 pixel 개수
        count_class_by_ROI = zeros(cutoff_color_cnt, numel(tissuenames));
    
        % imfuse 시 표시할 새하얀 (1 1 1) RGB 매트릭스를 생성해본다
        size_curImage_ROI_resize = size(curImage_ROI_resize);
        rgb_imfuse_roi = ones(size_curImage_ROI_resize(1), size_curImage_ROI_resize(2), 3);

        j=1;
        k=1;
        m=1;
        
        % 2022-03-02 : 확률 매트릭스를 가우스필터링 한 후 
        cnt_tot = 0;
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

            rgbout(j, k, :) = colors(max_class_idx(j,k),:);
    
            rgbout_TN(j, k, :) = colors_TN(max_class_idx(j,k),:);
    
            if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                continue
            end
            % if문으로 
            %max_class_TN(j, k) = sum((reshape(TUM_idx, [1 numel(class_num)]) & (temp_class_num(1) == class_num))) == 1;
            max_class_TN(j, k) = colors_TN_gray(max_class_idx(j,k));
            
            % 전체 샘플의 클래스별 pixel 개수 카운트
            count_class_by_ALL(1, max_class_idx(j, k)) = count_class_by_ALL(1, max_class_idx(j, k)) + 1;

            % ROI영역의 색상 개수별로 반복
            % ROI영역의 색상 개수별로 반복
            for color_i = 1:cutoff_color_cnt
                if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(color_i, :), [1 1 3])) == 3
                    % RGB 채널의 값이 모두 같으면(같은색상) 해당 ROI 영역색상별 클래스별 pixel 개수 카운트
                    count_class_by_ROI(color_i, max_class_idx(j, k)) = count_class_by_ROI(color_i, max_class_idx(j, k)) + 1;
                    rgb_imfuse_roi(j, k, :) = [1 0 1];
                end
            end
          end
        end
        
        %% 이미지 Plot 출력 
        if bShowPlot
            % tile 형태로 출력해본다
            figure();
            tiledlayout('flow');
            
            % H&E 6% 리사이즈 이미지
            if b_ExistPNG
                nexttile
                imshow(curImage_resize)
            end
            % ROI 이미지 원본
    %         nexttile
    %         imshow(curImage_ROI_org)
            % ROI 이미지를 red green 과 오버레이 시 구분하기 좋은 magenta 색상변경
            nexttile
            imshow(rgb_imfuse_roi);
            
            if b_ExistPNG
                % H&E 이미지 오버레이 ROI 이미지
                img_imfuse = imfuse(curImage_resize, curImage_ROI_org, 'blend');
                nexttile
                imshow(img_imfuse);
            end
            % RGB dotting 이미지
            nexttile
            imshow(rgbout)
    
            % RGB TN -> tumor(red) or Normal(green)    
            nexttile
            imshow(rgbout_TN)
            
            % Tumor dotting -> tumor (흰색) Normal (검정색) 배경도 검정색
            nexttile
            imshow(max_class_TN);
        %     title([mat_name, '-RGB_TN']);
    
            % RGB TN 오버레이 ROI(magenta)
            img_imfuse = imfuse(rgbout_TN, rgb_imfuse_roi, 'blend');
            nexttile
            imshow(img_imfuse);
        %     title([mat_name, '-imfuse-RGB-TN-ROI']);
            
            % Tumor dotting 오버레이 ROI(magenta)
            nexttile
            img_imfuse = imfuse(max_class_TN, rgb_imfuse_roi, 'blend');
            imshow(img_imfuse)
    
    %         print(gcf,'-dpng','-r600', strcat(output_path,'/sigma_', num2str(nSigma),'/', mat_name, '_', model_Rev, '_Sigma_', num2str(nSigma), '_scen_default.jpg'));
            print(gcf,'-dpng','-r600', strcat(output_path,'/', sample_name, '_Sigma_', num2str(nSigma), '_', str_filter_func, '.jpg'));
            close all;
        end

        %% 블러처리 전 분석상태에서 DICE 계수를 구해본다.
        % tot_sample_ALL : 전체 영역 샘플 pixel 개수 합계를 구한다.
        tot_sample_ALL = sum(count_class_by_ALL(1, TUM_idx|Normal_idx));
    
        % tot_tumor_ALL : 전체 영역 Tumor 클래스의 pixel 개수 합계를 구한다.
        tot_tumor_ALL = sum(count_class_by_ALL(1, TUM_idx));
        
        % tot_normal_ALL : 전체 영역 Normal 클래스의 pixel 개수 합계를 구한다.
        tot_normal_ALL = sum(count_class_by_ALL(1, Normal_idx));
        
        %% green 색상 쓰레스홀드 세팅별로 Blur 필터를 적용
         max_i=1
%         for max_i=1:numel(arr_green_max)
        for max_i=1:numel(arr_green_max)
            disp(strcat('max_i : ', num2str(max_i), " / ", num2str(numel(arr_green_max))));
            
            % allSummaryROIFiles 구조체 변수에 출력할 정보 세팅
            % slidename : 샘플 정보
            % mat_name  : 
            % color : ROI 색상정보
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).slide_name = allROIFiles(file_i).name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).mat_name = mat_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).color = cutoff_color_name(max_i);
            % tissuenames에 있는 클래스별로 구조체 필드를 생성한다.
            for n=1:numel(tissuenames)
                allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(strrep(string(tissuenames(n)), "-", "_")) = count_class_by_ALL(1, n);
            end
            
            for n=1:numel(tissuenames)
                allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(strcat(strrep(string(tissuenames(n)), "-", "_"), "_by_ROI")) = count_class_by_ROI(1, n);
            end
            % 이미지 필터를 해서 노이즈 제거해보기
            if (nSigma ~= 0)
                % 이미지 필터
                if str_filter_func == "imfilter"
                    nSigma_filter = ones(nSigma, nSigma);
                    nSigma_filter = nSigma_filter / (nSigma*nSigma);
                    rgbout_TN_filtered = imfilter(rgbout_TN, nSigma_filter, 'replicate','same');
                    max_class_TN_filtered = imfilter(max_class_TN, nSigma_filter, 'replicate', 'same');
                elseif str_filter_func == "imgaussfilt"
                    rgbout_TN_filtered = imgaussfilt(rgbout_TN, nSigma);
                    max_class_TN_filtered = imgaussfilt(max_class_TN, nSigma);
                end
            else
                rgbout_TN_filtered = rgbout_TN;
                max_class_TN_filtered = max_class_TN;
            end

% 중간 이미지 확인용 테스트 스크립트
% nexttile; imshow(max_class_TN)
% nexttile; imshow(max_class_TN_filtered )
% nexttile; imshow(max_class_TN_filtered_conv )

            % green_max : arr_green_max에 max_i번째 세팅값
            green_max = arr_green_max(max_i);
            %% 이미지 필터 후 필터 크기별 반복
            % count_class_TN_by_ALL : 블러 전체 샘플영역 클래스 카운트
            count_class_TN_by_ALL = zeros(1, numel(tissuenames_TN));
            % count_class_TN_by_ROI : curImage_ROI_resize에서 색칠한 색상 영역별 클래스 개수
            count_class_TN_by_ROI = zeros(1, numel(tissuenames_TN));
            % count_class_TN_by_REST : curImage_ROI_resize에서 색칠한 색상 영역 바깥 클래스 개수
            count_class_TN_by_REST = zeros(1, numel(tissuenames_TN));
            
            % ** 가우시안 필터별 * Green 색상 쓰레스홀드별 샘플 클래스 pixel 개수 카운트
            tot_sample_ALL_sigma = 0;
            tot_tumor_ALL_sigma = 0;
            tot_normal_ALL_sigma = 0;
            tot_tumor_by_ROI_sigma = 0;
            tot_normal_by_ROI_sigma = 0;
            tot_tumor_by_REST_sigma = 0;
            tot_normal_by_REST_sigma = 0;
            
            % ** 가우시안 필터별 * Green 색상 쓰레스홀드별 샘플 클래스 pixel 개수 카운트
            % 
            for j=1:mask_size(1)
                for k=1:mask_size(2)
                    % 2020-09-09 : 여기서 픽셀별로 jpg 색상이랑 비교해서 카운팅 해보자
                    % 제외할 클래스는 색상 컨트롤에서 제외함
                    if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                        max_class_TN_filtered(j, k) = colors_TN_gray(15);
                        continue
                    elseif max_class_TN_filtered(j, k) < green_max
                        % Normal인 경우 : green_max 값보다 값이 작다.
                        % 검정색으로 변경
                        max_class_TN_filtered(j, k) = 0.0;

                        % red 색상으로 이진화한다
                        rgbout_filtered(j, k, 1) = 1.0;
                        rgbout_filtered(j, k, 2) = 0.0;

                        % count_class_TN_by_ALL(1, 2)에 개수 카운트 + 1
                        count_class_TN_by_ALL(1, 2) = count_class_TN_by_ALL(1, 2) + 1;
                        % ROI 영역에 속한 픽셀인지 검사한다
                        % ROI 영역 안쪽이면 count_class_TN_by_ROI에 카운트 + 1
                        % ROI 영역 바깥쪽이면 count_class_TN_by_REST에 카운트 + 1
                        if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                            count_class_TN_by_ROI(1, 2) = count_class_TN_by_ROI(1, 2) + 1;
                        else
                            count_class_TN_by_REST(1, 2) = count_class_TN_by_REST(1, 2) + 1;
                        end
                    else
                        % Tumor인경우 : green_max 값보다 값이 크거나 같다
                        max_class_TN_filtered(j, k) = 1.0;
                        % Normal인 경우 : green 색상으로 이진화한다
                        rgbout_filtered(j, k, 1) = 0.0;
                        rgbout_filtered(j, k, 2) = 1.0;

                        % Normal인 경우 : count_class_TN_by_ALL(1, 1)에 개수 카운트 + 1
                        count_class_TN_by_ALL(1, 1) = count_class_TN_by_ALL(1, 1) + 1;

                        % ROI 영역에 속한 픽셀인지 검사한다
                        % ROI 영역 안쪽이면 count_class_TN_by_ROI에 카운트 + 1
                        % ROI  영역 바깥쪽이면 count_class_TN_by_REST에 카운트 + 1
                        if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                            count_class_TN_by_ROI(1, 1) = count_class_TN_by_ROI(1, 1) + 1;
                        else
                            count_class_TN_by_REST(1, 1) = count_class_TN_by_REST(1, 1) + 1;
                        end
                    end
                end
            end
            
            % tot_sample_ALL_sigma : 샘플의 전체 클래스 개수 합계를 구한다.
            tot_sample_ALL_sigma = sum(count_class_TN_by_ALL(1, :));

            % tot_tumor_ALL_sigma : 샘플의 Tumor 클래스의 합계를 구한다.
            tot_tumor_ALL_sigma = sum(count_class_TN_by_ALL(1, 1));

            % tot_normal_ALL_sigma : 샘플의 Normal 클래스의 합계를 구한다.
            tot_normal_ALL_sigma = sum(count_class_TN_by_ALL(1, 2));

            % tot_tumor_by_ROI_sigma : ROI 색상영역별 Tumor 클래스의 합계를 구한다.
            tot_tumor_by_ROI_sigma = sum(count_class_TN_by_ROI(1, 1));
            
            % tot_normal_by_ROI_sigma : ROI 색상영역별 Normal 클래스의 합계를 구한다.
            tot_normal_by_ROI_sigma = sum(count_class_TN_by_ROI(1, 2));

            % tot_by_ROI_sigma : ROI 영역 전체 픽셀 개수를 구한다.
            tot_by_ROI_sigma = sum(count_class_TN_by_ROI(1, :));

            % ROI 색상영역 바깥의 Tumor 합계를 구한다.
            tot_tumor_by_REST_sigma = sum(count_class_TN_by_REST(1, 1));

            % ROI 색상영역 바깥의 Normal 합계를 구한다.
            tot_normal_by_REST_sigma = sum(count_class_TN_by_REST(1, 2));

            % ROI 색상영역 바깥에 전체(제외할 클래스는 제외됨) 합계를 구한다.
            tot_by_REST_sigma = sum(count_class_TN_by_REST(1, :));
            
            % allSummaryROIFiles : 최종출력 엑셀파일에 카운팅한 픽셀개수를 저장하기 위함
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_ALL = tot_tumor_ALL_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_ALL = tot_normal_ALL_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_by_ROI = tot_tumor_by_ROI_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_by_ROI = tot_normal_by_ROI_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_by_REST = tot_tumor_by_REST_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_by_REST = tot_normal_by_REST_sigma;

            % blur 처리 후 ROI 색상영역 안 Tumor 다이스 계수
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_ROI_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_ROI_sigma * 2) / (tot_by_ROI_sigma + tot_tumor_ALL_sigma);

            % blur 처리 후 ROI 색상영역 *바깥* Normal 다이스 계수
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_ROI_normal_blur', num2str(nSigma)]) = (tot_normal_by_ROI_sigma * 2) / (tot_by_ROI_sigma + tot_normal_ALL_sigma);
            
            % blur 처리 후 ROI 색상영역 안 Tumor IoU 계수
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_ROI_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_ROI_sigma) / (tot_by_ROI_sigma + tot_tumor_ALL_sigma-tot_tumor_by_ROI_sigma);
            
            % blur 처리 후 ROI 색상영역 *바깥* Normal IoU 계수
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_ROI_normal_blur', num2str(nSigma)]) = (tot_normal_by_ROI_sigma) / (tot_by_ROI_sigma + tot_normal_ALL_sigma-tot_normal_by_ROI_sigma);
            
            % 2022-04-11 장은지 요청 : ROI영역 안에 Normal에 대한 다이스 계수와 ROI영역 바깥에 대한
            % Tumor 다이스 계수도 계산 요청
            % blur 처리한 후 ROI영역 바깥 다이스 계수 Tumor
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_REST_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_REST_sigma * 2) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_tumor_ALL_sigma);
            
            % blur 처리한 후 ROI영역 다이스 계수 Normal
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_REST_normal_blur', num2str(nSigma)]) = (tot_normal_by_REST_sigma * 2) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_normal_ALL_sigma);
            
            % blur 처리한 후 ROI영역 바깥 IoU 계수 Tumor
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_REST_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_REST_sigma) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_tumor_ALL_sigma-tot_tumor_by_ROI_sigma);
            
            % blur 처리한 후 ROI영역 IoU 계수 Normal
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_REST_normal_blur', num2str(nSigma)]) = (tot_normal_by_REST_sigma) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_normal_ALL_sigma-tot_normal_by_ROI_sigma);

            if bShowPlot
                figure();
                tiledlayout('flow');
                
                nexttile
                imshow(rgbout_TN);
                
                nexttile
                imshow(rgb_imfuse_roi);
                
                % 필터링한 RGB-TN 이미지를 cutoff로 분리한 rgbout_TN_filtered을 표시한다.
%                 nexttile
%                 imshow(rgbout_TN_filtered);

                % RGB TN 오버레이 ROI
%                 img_imfuse = imfuse(rgb_imfuse_roi, rgbout_TN_filtered, 'blend');  
%                 nexttile
%                 imshow(img_imfuse);

                nexttile
                imshow(max_class_TN_filtered);
                % 각 sub plot 별로 가우시안 필터 및 green 색상별로 컨트롤한 RGB dotting 이미지를 표시한다.
                img_imfuse = imfuse(rgb_imfuse_roi, max_class_TN_filtered, 'blend');  
                nexttile
                imshow(img_imfuse);
                print(gcf,'-dpng','-r600', strcat(output_path,'/', sample_name, '_filter_size_', num2str(nSigma), '_scen_', num2str(green_max*100), '_', str_filter_func, '.jpg'));
                close all;
            end
        end
    end
    % 필터 크기별로 엑셀파일을 출력한다.
    strToday = datestr(now,'yyyymmdd_HHMMSS');
    save_allSummaryROIFiles = struct2table(allSummaryROIFiles);
    writetable(save_allSummaryROIFiles, strcat(output_path, '/', model_Rev, '_', num2str(numel(allROIFiles)), '개샘플_ROI별_클래스분류_카운트_filter_size_', num2str(nSigma), '_', str_filter_func, '.xlsx'));
end