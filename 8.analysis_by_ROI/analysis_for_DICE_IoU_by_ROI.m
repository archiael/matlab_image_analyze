%% 2022-08-18 DICE�� IoU ��� ��� ��ũ��Ʈ ����
% <Input>
% 1. (�ʼ�) �м��� ���� �̸� ��� ��������(��������)
% 2. (�ʼ�) AI ��(mat)
% 3. (�ʼ�) ROI �̹��� ����(png)
% 4. (�ʼ�) AI �м��Ϸ� ����(mat)
% 5. (�ɼ�) H&E mini ������ �̹���

% <Output>
% 1. ���ú� Plot 1�� (jpg)
% 2. ���ú� Cutoff��(0~1, 0.1��) ROI ǥ�� �̹��� 11�� (jpg)
% 3. ���� ũ�⺰(arr_size_filter) �������� (xlsx), Input 1���� ���ð��� * cutoff ���� ��ŭ�� ���� ����

clear all, close all, format compact, clc

bShowPlot = true;

gpuDevice(2);

%% �м��� �ʿ��� ���� ����

% �м��� ���� ��� ���������� �о�´�.
sample_list = readtable("E:\git\AI_model_kimhyuntae\8.analysis_by_ROI\analysis_for_DICE_IoU_by_ROI.xlsx");

% ���� ��� ���� ���
disp(sample_list.Properties.VariableNames(:));
disp(numel(sample_list.Properties.VariableNames));

% cell�� string ������ ��ȯ
sample_list_string = string(sample_list.filename(:));
disp(numel(sample_list_string));

% RGB�� õ�� ��ȹ ǥ�� ����
% str_filter_func : imfilter(�������) / imgaussfilt(���콺 ��Ȱȭ ����)
str_filter_func = "imfilter";
% arr_size_filter = [2]; % ����þ� ���� ���ǽ� => 2*ceil(2*sigma)+1
arr_size_filter = [9]; % ��� ���� ����  9px�϶� 1mm�� ����

% ���� �� cutoff ������Ȧ�� ��ġ�� �񱳿�
% ���� cutoff ��ġ�� 0.5~0.6
arr_green_max = [0:0.1:1];

% 2022-07-19 ROI ������ ���� 1������ �Ǿ����� ROI ���� ������ �ݺ��� green cutoff������ ����
% cutoff_color_name : �ݺ��� cutoff ������ ����
cutoff_color_name = string(arr_green_max);
arr_green_max_cnt = numel(arr_green_max);

% cutoff_color_RGB : �ݺ��� cutoff ���� ��ŭ ������ ����
% Qu-Path 'tumor' ���� ROI ����
% �������� : 'tumor' ���� ���� �� ROI ������ [238, 178, 178]
cutoff_color_RGB = [255 178 178;];
cutoff_color_RGB_repmat = repmat(cutoff_color_RGB, numel(cutoff_color_name), 1);

% ROI�� ������ ���������� �Ʒ� ���ڸ� 1���� �ش� ���������� �����ؾ���
cutoff_color_cnt = 1;

%% �����̵� H&E tif  �̹��� ��� �ҷ����� (Option)
% ù��° main ��Ʈ
clear dir;
input_png_Path = 'Z:\SET_003_LEICA\2.resize_6%';
allPNGFiles = dir([input_png_Path,'/*.png']);

%% DICE�� IoU�� ����� ���� ��� �ҷ����� (�ʼ�)
% ROI ���� �̹��� ��� = DICE&IoU ����� ���ø��
clear dir;
% tuning 66 ����
% input_ROI_Path = 'Z:\SET_003_LEICA\5.ROI\5.2.analysis\2.tuning\1.ROI_svg_to_png';

% test 99 ����
input_ROI_Path = 'Z:\SET_003_LEICA\5.ROI\5.2.analysis\3.test\1.ROI_svg_to_png';

% TCGA surgical 187 ����
% input_ROI_Path = 'Z:\SET_013_TCGA\TCGA.STAD\5.ROI\2.analysis\20220811';

allROIFiles = dir([input_ROI_Path,'/*.png']);

%% TIF�� RGB_dotting�� �������(*.mat) ��� �ҷ�����
% TIF�� RGB_dotting�� ��� ����
% tuning
% inputPath = 'Z:\SET_003_LEICA\3.RGB_dotting\Rev0.01.13_LEICA_C04C05_tuning_66ea\mat';

% test
inputPath = 'Z:\SET_003_LEICA\3.RGB_dotting\Rev0.01.13_LEICA_C04C05_test_99ea\mat';

% TCGA
% inputPath = 'Z:\SET_013_TCGA\TCGA.STAD\3.RGB_dotting\Rev0.01.13_LEICA_C04C05\mat';
allMatFiles = dir([inputPath, '/*.mat']);

%% �м��� �ʿ��� ������ �� �ε� 
% �������� �ִ� ���� ���
model_path = 'Z:\DEEP_LEARNING_RGB_dotting\2.trained_model\Rev0.01.xx_SET_002_stomach_cancer_class_define\Rev0.01.13_LEICA_C04C05';
% ������ ��
model_file = 'tuning_C04C05_class_15_Rev0.01.13_LEICA_C04C05.mat';

% newData1 ������ ������ �ε�
newData1 = load('-mat', fullfile(model_path, model_file));

% newData1 struct ������ field�� ������ �� �ִ� ������
% �������� ���·� �޸𸮷� �ε�
vars = fieldnames(newData1);
for model_i = 1:length(vars)
    assignin('base', vars{model_i}, newData1.(vars{model_i}));
end
% myNet�� ������ �𵨿��� Input �̹��� ũ�⸦ �����´�.
% VGG19���� �⺻ 224x224x3 ũ����
layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize;

clearvars testing_tbl training_categories training_tbl vars newData1;

%% DICE & IoU ����� ����� ������ ���� ����
output_path = strcat(input_ROI_Path, '\DICE_IoU_���_����_', model_Rev);
mkdir(output_path);
winopen(output_path);

% tissuenames : ���� �з� ī�װ�
tissuenames = unique(myNet.Layers(end).ClassNames);

% ����(tumor) Ŭ���� one-hot ���ڵ����� ����
% C-04 = tumor ����
TUM_class = ["C-01" "C-02" "C-03" "C-04" "C-05"];
% �⺻�� false�� 15x1 �迭�� ����
TUM_idx = false(numel(tissuenames), 1);
% tissuenames ������ŭ ������� TUM_class�� ��ġ�ϴ� �ε����� true���� ����
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == TUM_class) > 0
        TUM_idx(var_i, 1) = true;
    end
end
% TUM_class�� ������ ������ Ȯ��
sum(TUM_idx)

% Others Ŭ����
others_class = ["C-13" "C-14"];
others_idx = false(numel(tissuenames), 1);
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == others_class) > 0
        others_idx(var_i, 1) = true;
    end
end
sum(others_idx)

% ���� �� �񺴺� ������ ����Ҷ� ������ 
% ��Ÿ Ŭ���� ���� (EDGE, RBC, BACK)
out_class = ["C-15"];
out_idx = false(numel(tissuenames), 1);
for var_i=1:numel(tissuenames)
    if sum(string(tissuenames(var_i)) == out_class) > 0
        out_idx(var_i, 1) = true;
    end
end
sum(out_idx)

% one-hot ���ڵ����� ����
Normal_idx = ~TUM_idx;
Normal_idx = Normal_idx & ~others_idx;
Normal_idx = Normal_idx & ~out_idx;
sum(Normal_idx)

% ��� ���ļ� tissuenames�� ������ ���ƾ� ��
disp(strcat("�� �з� Ŭ���� ���� : ", num2str(numel(tissuenames))));
disp(strcat("one-hot ���ڵ� ��ü Ŭ���� ���� : ", num2str(sum(TUM_idx|Normal_idx|others_idx|out_idx))));

% C-04�� tumor�� ������ tumor(red)�� �ƴϸ� normal(green)���� ����
% C-04 = tumor ����
if sum(TUM_class == 'C-04') == 1
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14          C-15     };
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
    colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
else % C-04 = normal ����
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14         C-15     };
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
    colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
end

% ����, �񺴺� 2���� ������ ī��Ʈ ����
% Others�� 
tissuenames_TN = {'Tumor', 'Normal', 'Others'};
class_num = 1:numel(tissuenames); % �з� Ŭ������ ���� �ε���

sigma_i = 1;

for sigma_i=1:numel(arr_size_filter)
    % sigma ���� ���������� ����ϱ����� �ʱ�ȭ
    clearvars allSummaryROIFiles;
    
    % ���͸� ����
    nSigma = arr_size_filter(sigma_i);

    file_i = 1;
    for file_i = 170:numel(sample_list_string)
        
        % 1. sample �̸��� �����Ѵ�.
        sample_name = strrep(sample_list_string(file_i), '-', '_');
        disp(strcat("filei : ", num2str(file_i), "  /  sample name : ", sample_name));
        

        % 2. ROI ���� ��Ͽ��� ���� �����̸��� ���� ROI ���� ���� üũ
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
            disp(strcat("������ ", num2str(file_i), "��° sample : ", sample_name, "�� ������ ROI ������ ã�� ���߽��ϴ�!!!!! ROI ���� ���θ� Ȯ���ϼ���"));
            return;
        else
            disp(strcat("ROI ������ ã�ҽ��ϴ� ROI_name : ", ROI_name));
        end
        % 2. ROI ���� ��Ͽ��� ���� �����̸��� ���� ROI ���� ã�� ����
        
        % 3. mat ���� ��Ͽ��� ���� �����̸��� ���� �м����(mat) ���� ���� üũ
        b_ExistMat = false;
        % mat_idx = 1;
        for mat_idx = 1:numel(allMatFiles)
            mat_name = strrep(allMatFiles(mat_idx).name, '-', '_');
            
            mat_path = [allMatFiles(mat_idx).folder,'/',allMatFiles(mat_idx).name];
            % sample_name�� ���Ե� mat������ ã�´�.
            is_equal_sample = strfind(upper(mat_name), upper(sample_name));
            if numel(is_equal_sample) == 1
                b_ExistMat = true;
                break
            end
        end
        if ~b_ExistMat
            disp(strcat("������ ", num2str(file_i), "��° sample : ", sample_name, "�� ������ mat ������ ã�� ���߽��ϴ�!!!!! mat ���� ���θ� Ȯ���ϼ���"));
            return;
        else
            disp(strcat("mat ������ ã�ҽ��ϴ� mat_name : ", mat_name));
        end
        
        % structure ���� newData1�� mat���Ͼȿ� ������ �������� �ҷ��´�
        newData1 = load('-mat', mat_path);
        % newData1.mask �������� base �۾������� mask ������ �ִ´�
        assignin('base', 'mask', newData1.('mask'));
        
        % mat���� ���鶧 ����� border
        % �⺻ [12 12] mat���Ͽ��� �ҷ��ͼ� �����
        border = [12 12];

        % newData1.mask �������� base �۾������� mask ������ �ִ´�
        try % ������ ������ mat���Ͽ� border�� ���� error�� ���°��� ���
            assignin('base', 'border', newData1.('border'));
        catch
        end
        disp(strcat("border : ", num2str(border(1)), ", ", num2str(border(2))))

        % newData1.model_Rev ���� ���� base �۾������� mask ������ �ִ´�
        assignin('base', 'model_Rev_by_mat', newData1.('model_Rev'));
        
        % mat���ϰ� AI �� Rev�� �������� üũ�Ͽ� �ȸ����� �����޽����� ��� ���� ROI�� �Ѿ��.
%         if string(model_Rev_by_mat) ~= string(model_Rev)
%             disp("mat������ ���� AI ������ AI �𵨰� ��ġ���� �ʽ��ϴ�!!!!!!!");
%             disp(strcat("�м��� �������� Rev :", model_Rev));
%             disp(strcat('mat ������ model Rev : ', model_Rev_by_mat'));
%             return;
%         end
         
        bsize = bsize(1:2)-2*border;
        % �е��� ������� �����ڸ� ���� ����
        rmov = ceil(border(1)/bsize(1));
        
        % ���� ������ȣ�� H&E mini ������ ���� ��Ͽ��� ROI_name�� ������ �̸� ���� üũ
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
            % H&E ������ �ҷ��´�.
            [curImage_org, map, alpha] = imread(PNG_path);
%       imshow(curImage_org) % H&E �̹����� Plotâ�� ���
        end
    
        % ROI ������ �ҷ��´�.
        [curImage_ROI_org, map, alpha] = imread(ROI_path);
%       imshow(curImage_ROI_org) % ROI �̹����� Plotâ�� ���

        curImage_ROI_org_size = size(curImage_ROI_org);
    
        %% RGB dotting�� ���� ����
        if rmov > 0
            mask_margin = mask((rmov):(end-rmov),(rmov):(end-rmov),:);
        else
            % rmov�� 0�ΰ�� �ε��� ������ ���� ó��
            mask_margin = mask((1):(end),(1):(end),:);
        end

        mask_size = size(mask_margin);
        
        % curImage_ROI_resize : õ������ ��ĥ�� �̹����� mask ����� �°� ������¡
        curImage_ROI_resize = imresize(curImage_ROI_org, mask_size(1:2));
        
        % curImage_resize : H&E �̹����� ROI ����� ���� ������¡
        if b_ExistPNG
            curImage_resize = imresize(curImage_org, curImage_ROI_org_size(1:2));
        end
        % activation ���� �̿��Ͽ� max�� Ŭ������ RGB �������� ��ȯ
        
        %% ���� ���� �������� ���� ū Ȯ������ ���� Ŭ������ RGB ������ ����ϰ� �ش� Ŭ������ ������ ī��Ʈ
        % max_class_idx : activation�� �ִ밪�� ã�� �ִ밪�� ���� �ε����� ������
        max_class_idx = zeros(mask_size(1), mask_size(2));
        
        % rgbout : activation�� �ִ밪�� Ŭ������ �ε����� ã�� colors(RGB)�� �ش� �ε��������� ��ȯ
        rgbout = zeros(mask_size(1), mask_size(2), 3);

        % Tumor class or Normal class ����(1 or 0) ����� �����Ͽ� 
        % ROI�� �󸶳� ��ġ���� ���θ� �����غ���
        max_class_TN = zeros(mask_size(1), mask_size(2));
        
        % rgbout_TN : Red(tumor) vs Green��(normal) RGB ���� ���� m x��Ʈ����
        rgbout_TN = zeros(mask_size(1), mask_size(2), 3); 
        
        % count_class_by_ALL : ��ü ���ÿ��� Ŭ������ pixel ����
        count_class_by_ALL = zeros(1, numel(tissuenames));

        % count_class_by_ROI : ROI �����ȿ� Ŭ������ pixel ����
        count_class_by_ROI = zeros(cutoff_color_cnt, numel(tissuenames));
    
        % imfuse �� ǥ���� ���Ͼ� (1 1 1) RGB ��Ʈ������ �����غ���
        size_curImage_ROI_resize = size(curImage_ROI_resize);
        rgb_imfuse_roi = ones(size_curImage_ROI_resize(1), size_curImage_ROI_resize(2), 3);

        j=1;
        k=1;
        m=1;
        
        % 2022-03-02 : Ȯ�� ��Ʈ������ ���콺���͸� �� �� 
        cnt_tot = 0;
        for j=1:mask_size(1)
          for k=1:mask_size(2)
            % 1. max(mask_margin(j,k,:)) : mask_margin(j, k)��° 15�� Ŭ������ ���� ū Ȯ����
            % 2. (mask_margin(j,k,:) == 1�������� ����ū Ȯ����) : Ȯ������ �ε��� logical ��
            % 3. class_num(���� ū Ȯ������ �ε��� logical ��)
            % ���� ū Ȯ������ �ε��� logical �� ����
            % temp_class_num = class_num([0 0 1 0 0 0 0 0 0 0 0 0 0 0 0])
            % temp_class_num = 3;
            temp_class_num = class_num(mask_margin(j,k,:)==max(mask_margin(j,k,:)));
            % ��Ȥ ������ Ȯ������ 2�� �̻��� ��� ù��° Ȯ���� ���� Ŭ������ ����
            % �� �߿��� Ŭ������ ���� Ŭ������ Ŭ���� ���ʿ� ��ġ�ϹǷ�
            max_class_idx(j, k) = temp_class_num(1);

            rgbout(j, k, :) = colors(max_class_idx(j,k),:);
    
            rgbout_TN(j, k, :) = colors_TN(max_class_idx(j,k),:);
    
            if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                continue
            end
            % if������ 
            %max_class_TN(j, k) = sum((reshape(TUM_idx, [1 numel(class_num)]) & (temp_class_num(1) == class_num))) == 1;
            max_class_TN(j, k) = colors_TN_gray(max_class_idx(j,k));
            
            % ��ü ������ Ŭ������ pixel ���� ī��Ʈ
            count_class_by_ALL(1, max_class_idx(j, k)) = count_class_by_ALL(1, max_class_idx(j, k)) + 1;

            % ROI������ ���� �������� �ݺ�
            % ROI������ ���� �������� �ݺ�
            for color_i = 1:cutoff_color_cnt
                if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(color_i, :), [1 1 3])) == 3
                    % RGB ä���� ���� ��� ������(��������) �ش� ROI �������� Ŭ������ pixel ���� ī��Ʈ
                    count_class_by_ROI(color_i, max_class_idx(j, k)) = count_class_by_ROI(color_i, max_class_idx(j, k)) + 1;
                    rgb_imfuse_roi(j, k, :) = [1 0 1];
                end
            end
          end
        end
        
        %% �̹��� Plot ��� 
        if bShowPlot
            % tile ���·� ����غ���
            figure();
            tiledlayout('flow');
            
            % H&E 6% �������� �̹���
            if b_ExistPNG
                nexttile
                imshow(curImage_resize)
            end
            % ROI �̹��� ����
    %         nexttile
    %         imshow(curImage_ROI_org)
            % ROI �̹����� red green �� �������� �� �����ϱ� ���� magenta ���󺯰�
            nexttile
            imshow(rgb_imfuse_roi);
            
            if b_ExistPNG
                % H&E �̹��� �������� ROI �̹���
                img_imfuse = imfuse(curImage_resize, curImage_ROI_org, 'blend');
                nexttile
                imshow(img_imfuse);
            end
            % RGB dotting �̹���
            nexttile
            imshow(rgbout)
    
            % RGB TN -> tumor(red) or Normal(green)    
            nexttile
            imshow(rgbout_TN)
            
            % Tumor dotting -> tumor (���) Normal (������) ��浵 ������
            nexttile
            imshow(max_class_TN);
        %     title([mat_name, '-RGB_TN']);
    
            % RGB TN �������� ROI(magenta)
            img_imfuse = imfuse(rgbout_TN, rgb_imfuse_roi, 'blend');
            nexttile
            imshow(img_imfuse);
        %     title([mat_name, '-imfuse-RGB-TN-ROI']);
            
            % Tumor dotting �������� ROI(magenta)
            nexttile
            img_imfuse = imfuse(max_class_TN, rgb_imfuse_roi, 'blend');
            imshow(img_imfuse)
    
    %         print(gcf,'-dpng','-r600', strcat(output_path,'/sigma_', num2str(nSigma),'/', mat_name, '_', model_Rev, '_Sigma_', num2str(nSigma), '_scen_default.jpg'));
            print(gcf,'-dpng','-r600', strcat(output_path,'/', sample_name, '_Sigma_', num2str(nSigma), '_', str_filter_func, '.jpg'));
            close all;
        end

        %% ��ó�� �� �м����¿��� DICE ����� ���غ���.
        % tot_sample_ALL : ��ü ���� ���� pixel ���� �հ踦 ���Ѵ�.
        tot_sample_ALL = sum(count_class_by_ALL(1, TUM_idx|Normal_idx));
    
        % tot_tumor_ALL : ��ü ���� Tumor Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_tumor_ALL = sum(count_class_by_ALL(1, TUM_idx));
        
        % tot_normal_ALL : ��ü ���� Normal Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_normal_ALL = sum(count_class_by_ALL(1, Normal_idx));
        
        %% green ���� ������Ȧ�� ���ú��� Blur ���͸� ����
         max_i=1
%         for max_i=1:numel(arr_green_max)
        for max_i=1:numel(arr_green_max)
            disp(strcat('max_i : ', num2str(max_i), " / ", num2str(numel(arr_green_max))));
            
            % allSummaryROIFiles ����ü ������ ����� ���� ����
            % slidename : ���� ����
            % mat_name  : 
            % color : ROI ��������
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).slide_name = allROIFiles(file_i).name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).mat_name = mat_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).color = cutoff_color_name(max_i);
            % tissuenames�� �ִ� Ŭ�������� ����ü �ʵ带 �����Ѵ�.
            for n=1:numel(tissuenames)
                allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(strrep(string(tissuenames(n)), "-", "_")) = count_class_by_ALL(1, n);
            end
            
            for n=1:numel(tissuenames)
                allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(strcat(strrep(string(tissuenames(n)), "-", "_"), "_by_ROI")) = count_class_by_ROI(1, n);
            end
            % �̹��� ���͸� �ؼ� ������ �����غ���
            if (nSigma ~= 0)
                % �̹��� ����
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

% �߰� �̹��� Ȯ�ο� �׽�Ʈ ��ũ��Ʈ
% nexttile; imshow(max_class_TN)
% nexttile; imshow(max_class_TN_filtered )
% nexttile; imshow(max_class_TN_filtered_conv )

            % green_max : arr_green_max�� max_i��° ���ð�
            green_max = arr_green_max(max_i);
            %% �̹��� ���� �� ���� ũ�⺰ �ݺ�
            % count_class_TN_by_ALL : �� ��ü ���ÿ��� Ŭ���� ī��Ʈ
            count_class_TN_by_ALL = zeros(1, numel(tissuenames_TN));
            % count_class_TN_by_ROI : curImage_ROI_resize���� ��ĥ�� ���� ������ Ŭ���� ����
            count_class_TN_by_ROI = zeros(1, numel(tissuenames_TN));
            % count_class_TN_by_REST : curImage_ROI_resize���� ��ĥ�� ���� ���� �ٱ� Ŭ���� ����
            count_class_TN_by_REST = zeros(1, numel(tissuenames_TN));
            
            % ** ����þ� ���ͺ� * Green ���� ������Ȧ�庰 ���� Ŭ���� pixel ���� ī��Ʈ
            tot_sample_ALL_sigma = 0;
            tot_tumor_ALL_sigma = 0;
            tot_normal_ALL_sigma = 0;
            tot_tumor_by_ROI_sigma = 0;
            tot_normal_by_ROI_sigma = 0;
            tot_tumor_by_REST_sigma = 0;
            tot_normal_by_REST_sigma = 0;
            
            % ** ����þ� ���ͺ� * Green ���� ������Ȧ�庰 ���� Ŭ���� pixel ���� ī��Ʈ
            % 
            for j=1:mask_size(1)
                for k=1:mask_size(2)
                    % 2020-09-09 : ���⼭ �ȼ����� jpg �����̶� ���ؼ� ī���� �غ���
                    % ������ Ŭ������ ���� ��Ʈ�ѿ��� ������
                    if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                        max_class_TN_filtered(j, k) = colors_TN_gray(15);
                        continue
                    elseif max_class_TN_filtered(j, k) < green_max
                        % Normal�� ��� : green_max ������ ���� �۴�.
                        % ���������� ����
                        max_class_TN_filtered(j, k) = 0.0;

                        % red �������� ����ȭ�Ѵ�
                        rgbout_filtered(j, k, 1) = 1.0;
                        rgbout_filtered(j, k, 2) = 0.0;

                        % count_class_TN_by_ALL(1, 2)�� ���� ī��Ʈ + 1
                        count_class_TN_by_ALL(1, 2) = count_class_TN_by_ALL(1, 2) + 1;
                        % ROI ������ ���� �ȼ����� �˻��Ѵ�
                        % ROI ���� �����̸� count_class_TN_by_ROI�� ī��Ʈ + 1
                        % ROI ���� �ٱ����̸� count_class_TN_by_REST�� ī��Ʈ + 1
                        if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                            count_class_TN_by_ROI(1, 2) = count_class_TN_by_ROI(1, 2) + 1;
                        else
                            count_class_TN_by_REST(1, 2) = count_class_TN_by_REST(1, 2) + 1;
                        end
                    else
                        % Tumor�ΰ�� : green_max ������ ���� ũ�ų� ����
                        max_class_TN_filtered(j, k) = 1.0;
                        % Normal�� ��� : green �������� ����ȭ�Ѵ�
                        rgbout_filtered(j, k, 1) = 0.0;
                        rgbout_filtered(j, k, 2) = 1.0;

                        % Normal�� ��� : count_class_TN_by_ALL(1, 1)�� ���� ī��Ʈ + 1
                        count_class_TN_by_ALL(1, 1) = count_class_TN_by_ALL(1, 1) + 1;

                        % ROI ������ ���� �ȼ����� �˻��Ѵ�
                        % ROI ���� �����̸� count_class_TN_by_ROI�� ī��Ʈ + 1
                        % ROI  ���� �ٱ����̸� count_class_TN_by_REST�� ī��Ʈ + 1
                        if sum(curImage_ROI_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                            count_class_TN_by_ROI(1, 1) = count_class_TN_by_ROI(1, 1) + 1;
                        else
                            count_class_TN_by_REST(1, 1) = count_class_TN_by_REST(1, 1) + 1;
                        end
                    end
                end
            end
            
            % tot_sample_ALL_sigma : ������ ��ü Ŭ���� ���� �հ踦 ���Ѵ�.
            tot_sample_ALL_sigma = sum(count_class_TN_by_ALL(1, :));

            % tot_tumor_ALL_sigma : ������ Tumor Ŭ������ �հ踦 ���Ѵ�.
            tot_tumor_ALL_sigma = sum(count_class_TN_by_ALL(1, 1));

            % tot_normal_ALL_sigma : ������ Normal Ŭ������ �հ踦 ���Ѵ�.
            tot_normal_ALL_sigma = sum(count_class_TN_by_ALL(1, 2));

            % tot_tumor_by_ROI_sigma : ROI ���󿵿��� Tumor Ŭ������ �հ踦 ���Ѵ�.
            tot_tumor_by_ROI_sigma = sum(count_class_TN_by_ROI(1, 1));
            
            % tot_normal_by_ROI_sigma : ROI ���󿵿��� Normal Ŭ������ �հ踦 ���Ѵ�.
            tot_normal_by_ROI_sigma = sum(count_class_TN_by_ROI(1, 2));

            % tot_by_ROI_sigma : ROI ���� ��ü �ȼ� ������ ���Ѵ�.
            tot_by_ROI_sigma = sum(count_class_TN_by_ROI(1, :));

            % ROI ���󿵿� �ٱ��� Tumor �հ踦 ���Ѵ�.
            tot_tumor_by_REST_sigma = sum(count_class_TN_by_REST(1, 1));

            % ROI ���󿵿� �ٱ��� Normal �հ踦 ���Ѵ�.
            tot_normal_by_REST_sigma = sum(count_class_TN_by_REST(1, 2));

            % ROI ���󿵿� �ٱ��� ��ü(������ Ŭ������ ���ܵ�) �հ踦 ���Ѵ�.
            tot_by_REST_sigma = sum(count_class_TN_by_REST(1, :));
            
            % allSummaryROIFiles : ������� �������Ͽ� ī������ �ȼ������� �����ϱ� ����
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_ALL = tot_tumor_ALL_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_ALL = tot_normal_ALL_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_by_ROI = tot_tumor_by_ROI_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_by_ROI = tot_normal_by_ROI_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_tumor_by_REST = tot_tumor_by_REST_sigma;
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).tot_normal_by_REST = tot_normal_by_REST_sigma;

            % blur ó�� �� ROI ���󿵿� �� Tumor ���̽� ���
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_ROI_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_ROI_sigma * 2) / (tot_by_ROI_sigma + tot_tumor_ALL_sigma);

            % blur ó�� �� ROI ���󿵿� *�ٱ�* Normal ���̽� ���
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_ROI_normal_blur', num2str(nSigma)]) = (tot_normal_by_ROI_sigma * 2) / (tot_by_ROI_sigma + tot_normal_ALL_sigma);
            
            % blur ó�� �� ROI ���󿵿� �� Tumor IoU ���
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_ROI_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_ROI_sigma) / (tot_by_ROI_sigma + tot_tumor_ALL_sigma-tot_tumor_by_ROI_sigma);
            
            % blur ó�� �� ROI ���󿵿� *�ٱ�* Normal IoU ���
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_ROI_normal_blur', num2str(nSigma)]) = (tot_normal_by_ROI_sigma) / (tot_by_ROI_sigma + tot_normal_ALL_sigma-tot_normal_by_ROI_sigma);
            
            % 2022-04-11 ������ ��û : ROI���� �ȿ� Normal�� ���� ���̽� ����� ROI���� �ٱ��� ����
            % Tumor ���̽� ����� ��� ��û
            % blur ó���� �� ROI���� �ٱ� ���̽� ��� Tumor
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_REST_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_REST_sigma * 2) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_tumor_ALL_sigma);
            
            % blur ó���� �� ROI���� ���̽� ��� Normal
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['DICE_REST_normal_blur', num2str(nSigma)]) = (tot_normal_by_REST_sigma * 2) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_normal_ALL_sigma);
            
            % blur ó���� �� ROI���� �ٱ� IoU ��� Tumor
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_REST_tumor_blur', num2str(nSigma)]) = (tot_tumor_by_REST_sigma) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_tumor_ALL_sigma-tot_tumor_by_ROI_sigma);
            
            % blur ó���� �� ROI���� IoU ��� Normal
            allSummaryROIFiles(max_i+(file_i-1)*arr_green_max_cnt).(['IoU_REST_normal_blur', num2str(nSigma)]) = (tot_normal_by_REST_sigma) / ((tot_sample_ALL_sigma - tot_by_ROI_sigma) + tot_normal_ALL_sigma-tot_normal_by_ROI_sigma);

            if bShowPlot
                figure();
                tiledlayout('flow');
                
                nexttile
                imshow(rgbout_TN);
                
                nexttile
                imshow(rgb_imfuse_roi);
                
                % ���͸��� RGB-TN �̹����� cutoff�� �и��� rgbout_TN_filtered�� ǥ���Ѵ�.
%                 nexttile
%                 imshow(rgbout_TN_filtered);

                % RGB TN �������� ROI
%                 img_imfuse = imfuse(rgb_imfuse_roi, rgbout_TN_filtered, 'blend');  
%                 nexttile
%                 imshow(img_imfuse);

                nexttile
                imshow(max_class_TN_filtered);
                % �� sub plot ���� ����þ� ���� �� green ���󺰷� ��Ʈ���� RGB dotting �̹����� ǥ���Ѵ�.
                img_imfuse = imfuse(rgb_imfuse_roi, max_class_TN_filtered, 'blend');  
                nexttile
                imshow(img_imfuse);
                print(gcf,'-dpng','-r600', strcat(output_path,'/', sample_name, '_filter_size_', num2str(nSigma), '_scen_', num2str(green_max*100), '_', str_filter_func, '.jpg'));
                close all;
            end
        end
    end
    % ���� ũ�⺰�� ���������� ����Ѵ�.
    strToday = datestr(now,'yyyymmdd_HHMMSS');
    save_allSummaryROIFiles = struct2table(allSummaryROIFiles);
    writetable(save_allSummaryROIFiles, strcat(output_path, '/', model_Rev, '_', num2str(numel(allROIFiles)), '������_ROI��_Ŭ�����з�_ī��Ʈ_filter_size_', num2str(nSigma), '_', str_filter_func, '.xlsx'));
end