%% 2023-04-12 ���� DICE�� dice()�� IoU�� jaccard()�� MATLAB �Լ��� �̿��Ͽ� ����ϴ� ��ũ��Ʈ ����
% AIM-154 : IoU ��� ��û (from. ������)
% <Input>
% 1. (�ʼ�) �м��� ���� �̸� ��� ��������(��������)
% 2. (�ʼ�) AI ��(mat) - RGB dotting �м��� �ߴ� AI ������
% 3. (�ʼ�) AI �м��Ϸ� ����(mat) - RGB dotting �м� ��� ����
% 4. (�ʼ�) ROI �̹��� ����(png)
% 5. (�ɼ�) H&E mini ������ �̹���

% <Output>
% 1. ���ð����� 4 by 4 �� 16���� plot 1�� (jpg)
% 2. ���ú� Cutoff��(0~1, 0.1��) ROI ǥ�� �̹��� 11�� (jpg)
% 3. ���� ũ�⺰(arr_size_filter) �������� (xlsx), Input 1���� ���ð��� * cutoff ���� ��ŭ�� ���� ����

clear all, close all, format compact, clc

% ���� �õ� ����
rng('default');
rng(2);

% �м� �� plot�� �������� ����
bShowPlot = false;

% �׷��� ī�� ����
gpuDevice(1);

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
% ���ʿ亯������
clearvars model_i testing_tbl training_categories training_tbl vars newData1;


% 2023-01-30 / ������ : ���� colors ���� 0~1���ΰ��� RGBǥ���� 0~255 ������ ��ȯ
colors = colors*255;

% myNet�� ������ �𵨿��� Input �̹��� ũ�⸦ �����´�.
% VGG19���� �⺻ 224x224x3 ũ����
layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize;

% tissuenames : ���� �з� ī�װ�
tissuenames = unique(myNet.Layers(end).ClassNames);

% AI�� �з��ϴ� Ŭ������ ���� �ε���
class_num = 1:numel(tissuenames);
disp("AI �� �ε� �Ϸ�")


%% �м��� �ʿ��� ���� ����

% 1. �м��� �����̸�(filename) �ʵ带 ���� �������� ���
str_Excel_File_Path = "Getting_DICE_IoU_from_ROI_sample_list - Copy.xlsx";

% 2. H&E �̴� �̹��� ��� (���� ���� �̹����� ROI�� ��� ǥ���ߴ��� Ȯ���ϱ� ����)
input_resize6_png_Path = 'Z:\2023_01_NPOGȮ���ӻ�\LEICA\����������� 100\2.resize_5%';

% 3. RGB �̹����� �����ϱ� ���� ������ �м��س��� RGB dotting ����� mat���� ����
% �м��س��� mat������ ���ٸ� RGB_dotting ��ũ��Ʈ�� �ش� H&E �̹����� �м��� �� mat���� ��θ� �����Ѵ�.
input_mat_Path = 'Z:\2023_01_NPOGȮ���ӻ�\LEICA\����������� 100\3.RGB_dotting\mat';

% 4. 
input_ROI_Path = 'Z:\2023_01_npogȮ���ӻ�\leica\����������� 100\5.roi\1.ejjang';


%% DICE & IoU ����� ����� ������ ���� ����
output_path = strcat(input_ROI_Path, '\Jaccard_ROI���̳ʸ�ȭ_���_����_', model_Rev);
mkdir(output_path);
winopen(output_path);

disp("Output ���� ���� �Ϸ�")

% ���� �з� ī�װ� : ����, �񺴺� Other 3���� ������ ī��Ʈ ����
tissuenames_category = {'Tumor', 'Normal', 'Others', 'Back'};

% �м��� ���� ��� ���������� �о�´�.
all_sample_list = readtable(str_Excel_File_Path);

% ���� ��� ���� ���
disp(all_sample_list.Properties.VariableNames(:));

% ����� ����� ���
disp(numel(all_sample_list.Properties.VariableNames));

% cell�� string ������ ��ȯ
all_sample_list_string = string(all_sample_list.filename(:));

% ���� ������ ���â�� ���
disp(numel(all_sample_list_string));
disp("�м��� ���� ����Ʈ �ε� �Ϸ�")


% str_filter_func : imfilter(�������) / imgaussfilt(���콺 ��Ȱȭ ����)
str_filter_func = "imfilter";
% arr_size_filter = [2]; % ����þ� ���� ���ǽ� => 2*ceil(2*sigma)+1
arr_size_filter = [9]; % ��� ���� ����  9px�϶� 1mm�� ����

% ���� �� cutoff ������Ȧ�� ��ġ�� �񱳿�
% ���� cutoff ��ġ�� 0.5~0.6
% arr_cutoff_max = [0:0.1:1];
% 2022-10-04 ����� : ��ũ��Ʈ ���� �� �ʹ� ���� ����� �����ϰ��� ������ �ƿ����� ������
% �� �� �ƿ������� �׽�Ʈ�Ϸ��� 117���� �ּ��� Ǯ� ���
% arr_cutoff_max = [0.5];
arr_cutoff_max = [128];

% 2022-07-19 ROI ������ ���� 1������ �Ǿ����� ROI ���� ������ �ݺ��� green cutoff������ ����
% cutoff_color_name : �ݺ��� cutoff ������ ����
cutoff_color_name = string(arr_cutoff_max);
arr_cutoff_max_cnt = numel(arr_cutoff_max);

% cutoff_color_RGB : �ݺ��� cutoff ���� ��ŭ ������ ����
% Qu-Path 'tumor' ���� ROI ����
% �������� : 'tumor' ���� ���� �� ROI ������ [238, 178, 178]
cutoff_color_RGB = [255 178 178;];
cutoff_color_RGB_repmat = repmat(cutoff_color_RGB, numel(cutoff_color_name), 1);

% ROI�� ������ ���������� �Ʒ� ���ڸ� 1���� �ش� ���������� �����ؾ���
cutoff_color_cnt = 1;

disp("���� ���� �Ϸ�")

%% �����̵� H&E tif  �̹��� ��� �ҷ����� (Option)
% ù��° main ��Ʈ
clear dir;
allHnEFiles = dir([input_resize6_png_Path,'/*.tif']);
disp(strcat("H&E ���� ", num2str(numel(allHnEFiles)), "�� �ε�"))

%% DICE�� IoU�� ����� ���� ��� �ҷ����� (�ʼ�)
% ROI ���� �̹��� ��� = DICE&IoU ����� ���ø��
clear dir;
allROIFiles = dir([input_ROI_Path,'/*.png']);
disp(strcat("ROI ���� ", num2str(numel(allROIFiles)), "�� �ε�"))

%% TIF�� RGB_dotting�� �������(*.mat) ��� �ҷ�����
% TIF�� RGB_dotting�� ��� ����
allMatFiles = dir([input_mat_Path, '/*.mat']);
disp(strcat("�м����(mat) ���� ", num2str(numel(allMatFiles)), "�� �ε�"))


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
%     colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
%     colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21; 155 21 21; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;];
    colors_TN_gray = [       255;       255;       255;       255;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;];
else % C-04 = normal ����
    %                       C-01       C-02       C-03      "C-04"      C-05     C-06     C-07     C-08     C-09     C-10     C-11     C-12         C-13        C-14         C-15     };
%     colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;]/255;
%     colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;]/255;
    colors_TN      = [ 155 21 21; 155 21 21; 155 21 21;   0 111 0; 155 21 21; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 0 111 0; 127 127 127; 127 127 127; 255 255 255;];
    colors_TN_gray = [       255;       255;       255;         0;       255;       0;       0;       0;       0;       0;       0;       0;         128;         128;        128;];
end

filter_size_i = 1;

for filter_size_i=1:numel(arr_size_filter)
    % sigma ���� ���������� ����ϱ����� �ʱ�ȭ
    clearvars allSummaryROIFiles;

    % ���͸� ����
    nFilterSize = arr_size_filter(filter_size_i);

    file_i = 1;
    for file_i = 1:numel(all_sample_list_string)

        % 1. sample �̸��� �����Ѵ�.
        sample_name = char(strrep(all_sample_list_string(file_i), '-', '_'));
%         sample_name = sample_name(1:10);

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

        if b_ExistROI
            disp(strcat("ROI ������ ã�ҽ��ϴ� ROI_name : ", ROI_name));
        else
            disp(strcat("������ ", num2str(file_i), "��° sample : ", sample_name, "�� ������ ROI ������ ã�� ���߽��ϴ�!!!!! ROI ���� ���θ� Ȯ���ϼ���"));
            return;
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
        if b_ExistMat
            disp(strcat("mat ������ ã�ҽ��ϴ� mat_name : ", mat_name));
        else
            disp(strcat("������ ", num2str(file_i), "��° sample : ", sample_name, "�� ������ mat ������ ã�� ���߽��ϴ�!!!!! mat ���� ���θ� Ȯ���ϼ���"));
            return;
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

        %% RGB dotting�� ���� ����
        if rmov > 0
            mask_margin = mask((rmov):(end-rmov),(rmov):(end-rmov),:);
        else
            % rmov�� 0�ΰ�� �ε��� ������ ���� ó��
            mask_margin = mask((1):(end),(1):(end),:);
        end
        % mask_margin�� Ȯ������ ��� �̹����� ��ȯ�ϴ°��� ���� ���� ����
        % mask_margin(1, 1, 1:15)
        %
        % for i = 1:164
        %     for j = 1:266
        %         for k=1:15
        %             a(k) = mask_margin(i, j, k)
        %             max_a = max(a);
        %             % a�� �ִ밪�� ������ �� ����
        %             a(a == max_a) % 0 0 0 0 0 0 0 ... 1
        %             % colors_TN�� 15��° ���� �����´�
        %                colors_TN(15, :)
        %                colors_TN(a == max_a)
        %     end
        % end
        
        mask_size = size(mask_margin);

        % ���� ������ȣ�� H&E mini ������ ���� ��Ͽ��� ROI_name�� ������ �̸� ���� üũ
        b_ExistPNG = false;
        % PNG_idx = 1;
        for PNG_idx = 1:numel(allHnEFiles)
            PNG_name = strrep(allHnEFiles(PNG_idx).name, '-', '_');

            PNG_path = [allHnEFiles(PNG_idx).folder,'/',allHnEFiles(PNG_idx).name];

            is_equal_sample = strfind(upper(PNG_name), upper(sample_name));
            if numel(is_equal_sample) == 1
                b_ExistPNG = true;
                break
            end
        end

        if b_ExistPNG
            disp(strcat("H&E ������ ã�ҽ��ϴ� PNG_name : ", PNG_name));
            % H&E ������ �ҷ��´�.
            try
                [curImage_org, map, alpha] = imread(PNG_path);
            catch
                file_name = replace(PNG_path, ".tif", ".png");
                [curImage_org, map, alpha] = imread(file_name);
            end
            %       imshow(curImage_org) % H&E �̹����� Plotâ�� ���
            % curImage_resize : H&E �̹����� ROI ����� ���� ������¡
            curImage_resize = imresize(curImage_org, mask_size(1:2));
        end

        % ROI ������ �ҷ��´�.
        [curImage_ROI_org, map, alpha] = imread(ROI_path);
        %       imshow(curImage_ROI_org) % ROI �̹����� Plotâ�� ���

        curImage_ROI_org_size = size(curImage_ROI_org);
        
        % 2022-08-23 from. ������
        % ROI ������ ��/��(Tumor)�� �����Ͽ� ������¡
        curImage_ROI_org_gray = im2gray(curImage_ROI_org);
        curImage_ROI_org_binary = imbinarize(curImage_ROI_org_gray, 'global');

% figure
% nexttile; imshow(curImage_ROI_org)
% nexttile; imshow(curImage_ROI_org_gray)
% nexttile; imshow(curImage_ROI_org_binary)
        %         curImage_ROI_org_binary_resize = imresize(curImage_ROI_org, mask_size(1:2));
        % "nearest" "bilinear" "bicubic"
        % curImage_ROI_org_binary_resize : õ������ ��ĥ�� �̹����� mask ����� �°� ������¡
        curImage_ROI_org_binary_resize = imresize(curImage_ROI_org_binary, mask_size(1:2), "bicubic");
% nexttile; imshow(curImage_ROI_org_binary_resize)
        % activation ���� �̿��Ͽ� max�� Ŭ������ RGB �������� ��ȯ
        
        %% ���� ���� ����Ȯ������ ���� ū Ȯ������ ���� Ŭ������ RGB ������ �������� �ش� Ŭ������ ������ ī��Ʈ
        % max_class_idx : activation�� �ִ밪�� ã�� �ִ밪�� ���� �ε����� ������
        max_class_idx = zeros(mask_size(1), mask_size(2));

        % rgbout : activation�� �ִ밪�� Ŭ������ �ε����� ã�� colors(RGB)�� �ش� �ε��������� ��ȯ
        rgbout = zeros(mask_size(1), mask_size(2), 3, 'uint8');

        % Tumor class or Normal class ����(1 or 0) ����� �����Ͽ�
        % ROI�� �󸶳� ��ġ���� ���θ� �����غ���
        colors_class_TN = zeros(mask_size(1), mask_size(2), 'uint8');

        % rgbout_TN : Red(tumor) vs Green��(normal) RGB ���� ���� m x��Ʈ����
        rgbout_TN = zeros(mask_size(1), mask_size(2), 3, 'uint8');

        % count_class_by_ALL : ��ü ���ÿ��� Ŭ������ pixel ����
        count_class_by_ALL = zeros(1, numel(tissuenames));

        % count_class_by_ROI : ROI �����ȿ� Ŭ������ pixel ����
        count_class_by_ROI = zeros(cutoff_color_cnt, numel(tissuenames));

        % imfuse �� ǥ���� ���Ͼ� (1 1 1) RGB ��Ʈ������ �����غ���
        size_curImage_ROI_resize = size(curImage_ROI_org_binary_resize);
        rgb_imfuse_roi = ones(size_curImage_ROI_resize(1), size_curImage_ROI_resize(2), 3);

        j=1;
        k=1;
        m=1;

        % 2022-03-02 : Ȯ�� ��Ʈ������ ���콺���͸� �� ��
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

                rgbout(j, k, 1:3) = colors(max_class_idx(j,k),1:3);

                rgbout_TN(j, k, 1:3) = colors_TN(max_class_idx(j,k),1:3);
                
                % out_idx�� ���� Ŭ������ ī��Ʈ���� �����Ѵ�.
                % 2023-02-01 / ������ : �˻꿡 �����ϵ��� ī��Ʈ�ϰ� �����Ѵ�.
%                 if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
%                     continue
%                 end

                % if������
                %max_class_TN(j, k) = sum((reshape(TUM_idx, [1 numel(class_num)]) & (temp_class_num(1) == class_num))) == 1;
                colors_class_TN(j, k) = colors_TN_gray(max_class_idx(j,k));

                % ��ü ������ Ŭ������ pixel ���� ī��Ʈ
                count_class_by_ALL(1, max_class_idx(j, k)) = count_class_by_ALL(1, max_class_idx(j, k)) + 1;

                % ROI������ ���� �������� �ݺ�
                % ROI������ ���� �������� �ݺ�
                for color_i = 1:cutoff_color_cnt
                    %                 if sum(curImage_ROI_org_binary_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(color_i, :), [1 1 3])) == 3
                    % 2022-08-23
                    if (curImage_ROI_org_binary_resize(j, k, :) == 0)
                        % RGB ä���� ���� ��� ������(��������) �ش� ROI �������� Ŭ������ pixel ���� ī��Ʈ
                        count_class_by_ROI(color_i, max_class_idx(j, k)) = count_class_by_ROI(color_i, max_class_idx(j, k)) + 1;
                        rgb_imfuse_roi(j, k, :) = [1 0 1];
                    end
                end
            end
        end

        %% �̹��� Plot ���
        if bShowPlot
            figure("Position", [0 0 1920 1080]);
%             tiledlayout('flow');
            %             whitebg([0 .5 .6])
            % 2023-02-06 / from. ������ : ���� ��� ���̾ƿ��� 2�������� ����
            subplot(4,4,1); imshow(curImage_resize); title(PNG_name);
            % 1. HE & ROI imfuse
            img_imfuse = imfuse(rgb_imfuse_roi, curImage_resize, 'blend');
            subplot(4,4,2); imshow(img_imfuse); title("WSI with ROI");
        end

        %% ��ó�� �� �м����¿��� DICE ����� ���غ���.
        % tot_sample_ALL : ��ü ���� ���� pixel ���� �հ踦 ���Ѵ�.
        tot_sample_ALL = sum(count_class_by_ALL(1, TUM_idx|Normal_idx|others_idx|out_idx));

        % tot_tumor_ALL : ��ü ���� Tumor Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_tumor_ALL = sum(count_class_by_ALL(1, TUM_idx));

        % tot_normal_ALL : ��ü ���� Normal Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_normal_ALL = sum(count_class_by_ALL(1, Normal_idx));

        % tot_others_ALL : ��ü ���� Other Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_other_ALL = sum(count_class_by_ALL(1, others_idx));
        
        % tot_out_ALL : ��ü ���� Background Ŭ������ pixel ���� �հ踦 ���Ѵ�.
        tot_out_ALL = sum(count_class_by_ALL(1, out_idx));

        % ī��Ʈ �˻� test
        %       disp(tot_sample_ALL)
        %       disp(sum([tot_tumor_ALL, tot_normal_ALL, tot_other_ALL, tot_out_ALL]))

        %% cutoff ���ú��� Blur ���͸� ����
        max_i=1;
        for max_i=1:numel(arr_cutoff_max)
            disp(strcat('max_i : ', num2str(max_i), " / ", num2str(numel(arr_cutoff_max))));

            % allSummaryROIFiles ����ü ������ ����� ���� ����
            % slidename : ���� ����
            % mat_name  :
            % color : ROI ��������
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).slide_name = sample_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).mat_name = mat_name;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).color = cutoff_color_name(max_i);

            % �̹��� ���͸� �ؼ� ������ ���� �ǽ�
            if (nFilterSize ~= 0)
                % �̹��� ����
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

            % % �߰� �̹��� Ȯ�ο� �׽�Ʈ ��ũ��Ʈ
            if bShowPlot
                subplot(4,4,3); imshow(rgbout_TN);                title("AI Analyze 3 colors");
                subplot(4,4,4); imshow(colors_class_TN);          title("AI Analyze grayscale");
                
                subplot(4,4,5); imshow(rgbout_TN_filtered);       title(strcat(str_filter_func, " AI Analyze 3 colors"));
                subplot(4,4,6); imshow(colors_class_TN_filtered); title(strcat(str_filter_func, " AI Analyze grayscale"));
                
            end
            % cutoff_max : arr_green_max�� max_i��° ���ð�
            cutoff_max = arr_cutoff_max(max_i);
            %% �̹��� ���� �� ���� ũ�⺰ �ݺ�
            % count_class_TN_by_ALL : �� ��ü ���ÿ��� Ŭ���� ī��Ʈ
            count_class_TN_by_ALL = zeros(1, numel(tissuenames_category));
            % count_class_TN_by_ROI : curImage_ROI_resize���� ��ĥ�� ���� ������ Ŭ���� ����
            count_class_TN_by_ROI = zeros(1, numel(tissuenames_category));
            % count_class_TN_by_REST : curImage_ROI_resize���� ��ĥ�� ���� ���� �ٱ� Ŭ���� ����
            count_class_TN_by_REST = zeros(1, numel(tissuenames_category));

            % ** ���ͺ� * Green ���� ������Ȧ�庰 ���� Ŭ���� pixel ���� ī��Ʈ
            j = 1;
            k = 1;

            % ī������ ���� ī�װ� ���� �ε����� ����
            % tissuenames_category�� ������ ��������
            n_category = 0;
            
            for j=1:mask_size(1)
                for k=1:mask_size(2)
                    % 2020-09-09 : ���⼭ �ȼ����� jpg �����̶� ���ؼ� ī���� �غ���
                    % ������ Ŭ������ ���� ��Ʈ�ѿ��� ������
                    if sum(max_class_idx(j, k) == class_num(out_idx)) > 0
                        %                         disp(strcat(num2str(j), "/", num2str(k), ":", num2str(max_class_idx(j, k))))
                        %                         continue % �˻��� ���� BACK�� ī��Ʈ �غ���
                        % Back ī�װ� ī��Ʈ
                        n_category = 4;

                        % other (ȸ��)���� ����ȭ
                        set_color_gray = colors_TN_gray(out_idx, :);
                        set_color_gray = set_color_gray(1); % ù��° ���� ���

                        % RGB_TN�� ������ other(ȸ��) �������� ����ȭ�Ѵ�
                        set_color_RGB = colors_TN(out_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % ù��° ���� ���

                    elseif sum(max_class_idx(j, k) == class_num(others_idx)) > 0
                        % other ī�װ� ī��Ʈ
                        n_category = 3;
                        % other (ȸ��)���� ����ȭ
                        set_color_gray = colors_TN_gray(others_idx, :);
                        set_color_gray = set_color_gray(1); % ù��° ���� ���

                        % RGB_TN�� ������ other(ȸ��) �������� ����ȭ�Ѵ�
                        set_color_RGB = colors_TN(others_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % ù��° ���� ���

                    elseif colors_class_TN_filtered(j, k) >= cutoff_max
                        % ������ ��� �̹����� ��ϰ��� cutoff ������ ���� ���ų� ũ�ٸ�
                        % ī������ ���� ī�װ��� Tumor
                        % tissuenames_category�� 2��°�� Tumor
                        n_category = 1;

                        % Tumor(���)���� ����ȭ
                        set_color_gray = colors_TN_gray(TUM_idx, :);
                        set_color_gray = set_color_gray(1); % ù��° ���� ���

                        % RGB_TN�� ������ normal(�׸�) �������� ����ȭ�Ѵ�
                        set_color_RGB = colors_TN(TUM_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % ù��° ���� ���

                    elseif colors_class_TN_filtered(j, k) < cutoff_max
                        % ������ ��� �̹����� ��ϰ��� cutoff ������ ���� �۴ٸ�
                        % ī������ ���� ī�װ��� Normal
                        % tissuenames_category�� 2��°�� Normal
                        n_category = 2;

                        % normal(������)���� ����ȭ
                        set_color_gray = colors_TN_gray(Normal_idx, :);
                        set_color_gray = set_color_gray(1); % ù��° ���� ���

                        % RGB_TN�� ������ normal(�׸�) �������� ����ȭ�Ѵ�
                        set_color_RGB = colors_TN(Normal_idx, :);
                        set_color_RGB = set_color_RGB(1, :); % ù��° ���� ���
                    end

                    % ��ó���� colors_class_TN_filtered �̹����� ����ȭ�Ѵ�.
                    colors_class_TN_filtered(j, k) = set_color_gray;

                    % ��ó���� rgbout_TN_filtered �̹����� ����ȭ�Ѵ�.
                    rgbout_TN_filtered(j, k, :) = set_color_RGB;

                    % ��ü������ Normal ī��Ʈ + 1
                    count_class_TN_by_ALL(1, n_category) = count_class_TN_by_ALL(1, n_category) + 1;
                    % ROI ������ ���� �ȼ����� �˻��Ѵ�
                    % ROI ���� �����̸� count_class_TN_by_ROI�� ī��Ʈ + 1
                    % ROI ���� �ٱ����̸� count_class_TN_by_REST�� ī��Ʈ + 1
                    %                     if sum(curImage_ROI_org_binary_resize(j, k, :) == reshape(cutoff_color_RGB_repmat(max_i, :), [1 1 3])) == 3
                    if (curImage_ROI_org_binary_resize(j, k, :) == 0)
                        count_class_TN_by_ROI(1, n_category) = count_class_TN_by_ROI(1, n_category) + 1;
                    else
                        count_class_TN_by_REST(1, n_category) = count_class_TN_by_REST(1, n_category) + 1;
                    end
                end
            end

            % �߰����� : ���ͷ� ������ ������ ī�װ����� ������ ��ó���� �̹��� ǥ��
            if bShowPlot
                subplot(4,4,7); imshow(rgbout_TN_filtered);       title(strcat("post processing", " AI Analyze 3colors"));
                subplot(4,4,8); imshow(colors_class_TN_filtered); title(strcat("post processing", " AI Analyze grayscale"));
            end

            % ��ü ���� �ȼ� ī��Ʈ ����
            % cnt_all_sum : ������ ��ü Ŭ���� ���� �հ踦 ���Ѵ�.
            cnt_all_sum = sum(count_class_TN_by_ALL(1, :));
            % cnt_all_tumor : ������ Tumor Ŭ������ �հ踦 ���Ѵ�.
            cnt_all_tumor = sum(count_class_TN_by_ALL(1, 1));
            % cnt_all_normal : ������ Normal Ŭ������ �հ踦 ���Ѵ�.
            cnt_all_normal = sum(count_class_TN_by_ALL(1, 2));
            % cnt_all_others : ������ Normal Ŭ������ �հ踦 ���Ѵ�.
            cnt_all_others = sum(count_class_TN_by_ALL(1, 3));
            % cnt_all_back : ������ Normal Ŭ������ �հ踦 ���Ѵ�.
            cnt_all_back = sum(count_class_TN_by_ALL(1, 4));


            % ROI ���� �ȼ� ī��Ʈ ����
            %             % cnt_ROI_sum : ROI ���� Tumor�� Normal�� Other, Backgroud�� ��ü �ȼ� ������ ���Ѵ�.
            cnt_ROI_sum = sum(count_class_TN_by_ROI(1, :));

% �˻��
% disp( cnt_ROI_sum == sum(sum(curImage_ROI_org_binary_resize==false)) )

            % cnt_ROI_tumor : ROI �����ȿ� Tumor Ŭ������ �հ踦 ���Ѵ�.
            cnt_ROI_tumor = sum(count_class_TN_by_ROI(1, 1));
            % cnt_ROI_normal : ROI �����ȿ� Normal Ŭ������ �հ踦 ���Ѵ�.
            cnt_ROI_normal = sum(count_class_TN_by_ROI(1, 2));
            % cnt_ROI_others : ������ Others Ŭ������ �հ踦 ���Ѵ�.
            cnt_ROI_others = sum(count_class_TN_by_ROI(1, 3));
            % cnt_ROI_back : ������ Back Ŭ������ �հ踦 ���Ѵ�.
            cnt_ROI_back = sum(count_class_TN_by_ROI(1, 4));

% �˻��
% disp(strcat("ROI���� ��ü �ȼ� ���� cnt_ROI_sum : ", num2str(cnt_ROI_sum)))
% disp(strcat("ROI���� tumor �ȼ� ���� �� C-01~C05 : ", num2str(sum(count_class_TN_by_ROI(1, 1:4)))))
% disp(strcat("ROI���� tumor �ȼ� ���� cnt_ROI_tumor : ", num2str(cnt_ROI_tumor)))
% disp(strcat("ROI���� tumor �ȼ� ���� �� C-01~C05 : ", num2str(sum(count_class_TN_by_ROI(1, 1)))))
% disp(strcat("ROI���� normal �ȼ� ���� cnt_ROI_normal : ", num2str(cnt_ROI_normal)))
% disp(strcat("ROI���� normal �ȼ� ���� �� C-06~C12 : ", num2str(sum(count_class_TN_by_ROI(1, 2)))))
% disp(strcat("ROI���� others �ȼ� ���� cnt_ROI_others : ", num2str(cnt_ROI_others)))
% disp(strcat("ROI���� others �ȼ� ���� �� C-13~C14 : ", num2str(sum(count_class_TN_by_ROI(1, 3)))))
% disp(strcat("ROI���� back �ȼ� ���� cnt_ROI_back : ", num2str(cnt_ROI_back)))
% disp(strcat("ROI���� back �ȼ� ���� �� C-15 : ",     num2str(sum(count_class_TN_by_ROI(1, 4)))))

            % REST ���� �ȼ� ī��Ʈ ����
            % ROI ���󿵿� �ٱ��� ��ü(other�� background ���ܵ�) �հ踦 ���Ѵ�.
            cnt_REST_sum = sum(count_class_TN_by_REST(1, :));

% �˻��
% disp( cnt_REST_sum == sum(sum(curImage_ROI_org_binary_resize==true)) )

            % cnt_REST_tumor : ROI ���� �ٱ��� Tumor �հ踦 ���Ѵ�.
            cnt_REST_tumor = sum(count_class_TN_by_REST(1, 1));
            % cnt_REST_normal : ROI ���� �ٱ��� Normal �հ踦 ���Ѵ�.
            cnt_REST_normal = sum(count_class_TN_by_REST(1, 2));
            % cnt_REST_others : ROI ���� �ٱ��� Others �հ踦 ���Ѵ�.
            cnt_REST_others = sum(count_class_TN_by_REST(1, 3));
            % cnt_REST_back : ROI ���� �ٱ��� Background �հ踦 ���Ѵ�.
            cnt_REST_back = sum(count_class_TN_by_REST(1, 4));
            

% �˻� Test
% disp(string(cnt_all_sum == cnt_ROI_sum + cnt_REST_sum))

            %% A. blur ó�� �� *ROI���� ��* + *Tumor* ����� ���ϱ� ���� ���� ����
%             TP = cnt_ROI_tumor;                   % ROI ���󿵿� ������(Positive) Tumor(Positive) Ŭ������ �հ踦 ���Ѵ�. 
%             FP = cnt_REST_tumor;                  % ROI ���󿵿� �ٱ���(Negative) Tumor(Positive) �հ踦 ���Ѵ�.
%             FN = cnt_ROI_normal+cnt_ROI_others;   % ROI ���󿵿� ������(Positive) Normal(Negative) Ŭ������ �հ踦 ���Ѵ�.
%             TN = cnt_REST_normal+cnt_REST_others; % ROI ���󿵿� �ٱ���(Negative) Normal(Negative) �հ踦 ���Ѵ�.

            %% A-1. *ROI���� ��* DICE ��� �˻��� ���� ����
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
            % AI�� tumor�� ������ ���� true
            AI_BW_tumor = colors_class_TN_filtered == 255;
            % �������� tumor�� ������ ���� true
            Doctor_BW_tumor = ~curImage_ROI_org_binary_resize;
            % MATLAB ���� dice() �Լ��� DICE ���
            DICE_ROI_tumor = dice(AI_BW_tumor, Doctor_BW_tumor);

            % dice ROI Tumor ��� ����
            if bShowPlot
                subplot(4,4,9); imshow(Doctor_BW_tumor);                   title(strcat('Doctor_BW_tumor'), 'Interpreter', 'none');
                subplot(4,4,10); imshow(AI_BW_tumor);                         title(strcat('AI_BW_tumor'), 'Interpreter', 'none');
                subplot(4,4,11); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('DICE_ROI_tumor : ', num2str(round(DICE_ROI_tumor, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['DICE_ROI_tumor', num2str(nFilterSize)]) = DICE_ROI_tumor;
            

            %% A-2. *ROI���� ��* IoU �˻��� ���� ����
%             % �������->       TP                /    ((TP + FN)            + (TP + FP)              - TP)
%                             4714                 /    ((4714 + (287 + 360)) + (4714+1644)            - 4714)
%                             4714                 /    (5361                 + 6358                   - 4714)
%                       cnt_ROI_tumor              / (cnt_ROI_sum       + cnt_all_tumor    - cnt_ROI_tumor)
%             % blur ó�� �� ROI ���󿵿� �� Tumor IoU ���
%             IoU_ROI_tumor = (cnt_ROI_tumor) / (cnt_ROI_sum + cnt_all_tumor-cnt_ROI_tumor);            

%           % ������ ���->    TP                  /     (TP                  + FP                      + FN)
%                             4714                 /     (4714                + 1644                    + (287 + 360))
%                         cnt_ROI_tumor            / (cnt_ROI_tumor + cnt_REST_tumor + cnt_ROI_normal + cnt_ROI_others)
%             IoU_ROI_tumor = cnt_ROI_tumor / (cnt_ROI_tumor + cnt_REST_tumor + cnt_ROI_normal + cnt_ROI_others)
            
            % MATLAB ���� jaccard() �Լ��� IoU ���
            IoU_ROI_tumor = jaccard(AI_BW_tumor, Doctor_BW_tumor);

            % IoU ROI Tumor ��� ����
            if bShowPlot
                subplot(4,4,12); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('IoU_ROI_tumor : ', num2str(round(IoU_ROI_tumor, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['IoU_ROI_tumor', num2str(nFilterSize)]) = IoU_ROI_tumor;
            
            %% B. blur ó�� �� *(������ normal ROI = tumor ROI�� �ݴ�)* + *Normal* ����� ���ϱ� ���� ���� ����
% TN = cnt_ROI_tumor+cnt_ROI_others;   % (TP->FN) ROI ���󿵿� ������(Positive) Tumor(Negative) Ŭ������ �հ踦 ���Ѵ�. 
% FN = cnt_REST_tumor+cnt_REST_others; % (FP->TN) ROI ���󿵿� �ٱ���(Negative) Tumor(Negative) �հ踦 ���Ѵ�.
% FP = cnt_ROI_normal;                 % (FN->FP) ROI ���󿵿� ������(Positive) Normal(Positive) Ŭ������ �հ踦 ���Ѵ�.
% TP = cnt_REST_normal;                % (TN->TP) ROI ���󿵿� �ٱ���(Negative) Normal(Positive) �հ踦 ���Ѵ�.
            %% B-1. *(������ normal ROI = tumor ROI�� �ݴ�)* DICE ��� �˻��� ���� ����
%             similarity = 2 * TP                / ( 2 * TP + FP + FN)
%                               2 * TP           / ( 2 * TP                + FP             + FN)
%             similarity = (2 * cnt_REST_normal) / ((2 * cnt_REST_normal) + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others))

            % AI�� normal�� ������ ���� (true)
            AI_BW_normal = colors_class_TN_filtered == 0;
            
            % AI�� background �� ������ ���� (true)
            AI_BW_back = (colors_class_TN_filtered == 128);
            % �������� normal�� ������ ������ �������� ����
            % ���� ROI�� �����Ѵ��� AI�� back�׶���� ������ ������ falseó��
            % ��� ����(tumor = false, normal = true, back = false)            
            Doctor_BW_normal = xor(curImage_ROI_org_binary_resize, AI_BW_back);
%             Doctor_BW_normal = curImage_ROI_org_binary_resize == ~AI_BW_back;
            DICE_ROI_normal = dice(AI_BW_normal, Doctor_BW_normal);

% dice ��� ����
% IoU ROI Tumor ��� ����
            if bShowPlot
                subplot(4,4,13); imshow(Doctor_BW_normal);                       title(strcat('Doctor_BW_normal(if)'), 'Interpreter', 'none');
                subplot(4,4,14); imshow(AI_BW_normal);                           title(strcat('AI_BW_normal'), 'Interpreter', 'none');
                subplot(4,4,15); imshow(imfuse(AI_BW_normal, Doctor_BW_normal)); title(strcat('DICE_ROI_normal : ', num2str(round(DICE_ROI_normal, 3))), 'Interpreter', 'none');
            end

            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['DICE_ROI_normal', num2str(nFilterSize)]) = DICE_ROI_normal;
            
            %% B-2. *(������ normal ROI = tumor ROI�� �ݴ�)* IoU ��� �˻��� ���� ����
            % blur ó�� �� ROI ���󿵿� Normal IoU ���
            % similarity =    TP                /  (TP                      + FP                       + FN)
            %             cnt_REST_normal / (cnt_REST_normal + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others))            
%             IoU_ROI_normal = cnt_REST_normal / (cnt_REST_normal + cnt_ROI_normal + (cnt_REST_tumor+cnt_REST_others));
            % MATLAB ���� jaccard() �Լ��� IoU ���
            IoU_ROI_normal = jaccard(AI_BW_normal, Doctor_BW_normal);
            % IoU ROI Tumor ��� ����
            if bShowPlot
                subplot(4,4,16); imshow(imfuse(AI_BW_normal, Doctor_BW_normal)); title(strcat('IoU_ROI_normal : ', num2str(round(IoU_ROI_normal, 3))), 'Interpreter', 'none');
            end
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).(['IoU_ROI_normal', num2str(nFilterSize)]) = IoU_ROI_normal;

            % allSummaryROIFiles : ������� �������Ͽ� ī������ �ȼ������� �����ϱ� ����
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_SUM    = cnt_all_sum;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_tumor  = cnt_all_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_normal = cnt_all_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_others = cnt_all_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ALL_back   = cnt_all_back;
            
            % �˻�� �׸� �߰�
            % tot_tumor_normal_SUM �׸�� tot_ROI_REST_SUM �׸��� ���ڰ� ���ƾ� �Ѵ�
%             allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).tot_tumor_normal_SUM = cnt_all_tumor + cnt_all_normal;
%             allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).tot_ROI_REST_SUM = cnt_ROI_tumor + cnt_ROI_normal + cnt_REST_tumor + cnt_REST_normal;
            % �˻�� �׸� �߰�

            % ROI ����
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_tumor   = cnt_ROI_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_normal  = cnt_ROI_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_others  = cnt_ROI_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).ROI_back    = cnt_ROI_back;
            
            % ROI �ٱ� ����
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_tumor  = cnt_REST_tumor;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_normal = cnt_REST_normal;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_others = cnt_REST_others;
            allSummaryROIFiles(max_i+(file_i-1)*arr_cutoff_max_cnt).REST_back   = cnt_REST_back;

            if bShowPlot
                print(gcf,'-djpeg','-r300', strcat(output_path,'/', sample_name, '_1p', '_filter_size_', num2str(nFilterSize), '_', str_filter_func, '.jpg'));
                
                close all;
                % 2023-02-06 : ������ ��û �系Ȯ�ο� 2������ �߰�
                figure("Position", [0 0 1920 1080]);
    %             tiledlayout('flow');
                %             whitebg([0 .5 .6])
                % 2023-02-06 / from. ������ : ���� ��� ���̾ƿ��� 2�������� ����
                subplot(2,3,1); imshow(img_imfuse); title("Pathologist tumor");
    %             subplot(4,4,9); imshow(Doctor_BW_tumor);                   title(strcat('Doctor_BW_tumor'), 'Interpreter', 'none');
                subplot(2,3,2); imshow(Doctor_BW_tumor); title("Pathologist tumor BW");
    %             subplot(4,4,11); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('DICE_ROI_tumor : ', num2str(round(DICE_ROI_tumor, 3))), 'Interpreter', 'none');
                subplot(2,3,3); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('Dice tumor : ', num2str(round(DICE_ROI_tumor, 3))), 'Interpreter', 'none');
    %             subplot(4,4,8); imshow(colors_class_TN_filtered); title(strcat("post processing", " AI Analyze grayscale"));
                subplot(2,3,4); imshow(colors_class_TN_filtered); title("AI tumor");
    %             subplot(4,4,10); imshow(AI_BW_tumor);                         title(strcat('AI_BW_tumor'), 'Interpreter', 'none');
                subplot(2,3,5); imshow(AI_BW_tumor); title("AI tumor BW");
    %             subplot(4,4,12); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('IoU_ROI_tumor : ', num2str(round(IoU_ROI_tumor, 3))), 'Interpreter', 'none');
                subplot(2,3,6); imshow(imfuse(AI_BW_tumor, Doctor_BW_tumor)); title(strcat('IoU tumor : ', num2str(round(IoU_ROI_tumor, 3))), 'Interpreter', 'none');
                print(gcf,'-djpeg','-r300', strcat(output_path,'/', sample_name, '_2p' ,'_filter_size_', num2str(nFilterSize), '_', str_filter_func, '.jpg'));
            end
        end
        close all;
    end
    % ���� ũ�⺰�� ���������� ����Ѵ�.
    strToday = datestr(now,'yyyymmdd_HHMM');
    save_allSummaryROIFiles = struct2table(allSummaryROIFiles);
    writetable(save_allSummaryROIFiles, strcat(output_path, '/', model_Rev, '_', num2str(numel(allROIFiles)), '������_ROI��_Ŭ�����з�_ī��Ʈ_filter_size_', num2str(nFilterSize), '_', str_filter_func, '_', strToday,'.xlsx'));
end