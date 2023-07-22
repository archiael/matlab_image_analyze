% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "A deep learning based stroma score is an independent prognostic 
% factor in colorectal cancer"
% Please refer to the article and the supplemntary material for a
% detailed description of the procedures. This is experimental software
% and should be used with caution.
% 
% this script will create a deep dream visualization
%
% assumes the neural net is loaded as myNet, works for VGG19

clear all, close all, format compact, clc

gpuDevice(2);

%% Load VGG19 Model
%model_path = 'E:\deep_stroma_score\2.Stomach_cancer_training\2.trained_model\Rev0.01.xx_SET_002_stomach_cancer_class_define\Rev0.01.11_class18_ver3';

model_path = 'C:\2-2.test_TESTSET_ver_02_result';
model_file = ['\', 'sub_1_lastNet_TEXTURE_VGG19_class_15_Rev0.01.08_15class_accuracy_0.87248.mat'];
newData1 = load('-mat', [model_path, model_file]);
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end
clearvars newData1 vars;

output_dir = ['C:\train_VGG19\2.trained_model\Rev0.01.xx_history\Rev0.01.12_LEICA_vgg19_learn_0.0003_sgdm_final/deepDreamImage'];
mkdir(output_dir);

% model_Rev = 'CNN_Regression_origin';
% ������ ���̾��� �н��� Ư¡�� �ð�ȭ

% 'PyramidLevels' ? �Ƕ�̵� �ܰ��� ����
% ��� ������ �����ϴ� �� ���� ���� �ػ� ���� �Ƕ�̵� �ܰ��� ������, ���� ������ �����˴ϴ�. 
% ���귮�� �ø��� ��� �� ū ��� ������ �����Ϸ��� �Ƕ�̵� �ܰ��� ������ �ø��ʽÿ�. 
% �ʱ� ����� ũ�Ⱑ ���� ������ �����Ϸ��� �ܰ��� ������ 1�� �����Ͻʽÿ�.
% 2021-03-10 : 5�� �����ϴ°� ���� ������ ����Ƽ�� Ư¡�� ǥ�������� Ȯ��
arr_pyram = [5]
% 'NumIterations' ? �Ƕ�̵� �ܰ�� �ݺ� Ƚ��
% 10 (����Ʈ ��) | ���� ����
% �Ƕ�̵� �ܰ�� �ݺ� Ƚ����, ���� ������ �����˴ϴ�. ���귮�� �ø��� ��� �� ���� ������ �����Ϸ��� �ݺ� Ƚ���� �ø��ʽÿ�.
% 2021-03-09 : 75�� ���� ������ ����Ƽ�� Ư¡�� ǥ�������� Ȯ��
arr_iter = [75]
% 'PyramidScale' ? �Ƕ�̵� �ܰ� �� ������
%1.4 (����Ʈ ��) | 1���� ū ��Į�� ��
%�� �Ƕ�̵� �ܰ� ���� �����Ϸ�, 1���� ū ��Į�� ������ �����˴ϴ�. 
% ��� ���� �̼��� ���� ������ �ݿ��Ϸ��� �Ƕ�̵� �������� ���̽ʽÿ�. 
% �Ƕ�̵� �������� �����ϸ� �Ű�� ���� �κп� �ִ� �������� ������ ���� ������ �����ϴ� �� ������ �˴ϴ�.
% 2021-03-10 : 1.1�� �������� Ư¡���� ��°����� ������
arr_scale = [1.1]
%analyzeNetwork(net);
%classes = myNet.Layers(end).Classes;
%channels = 1:numel(myNet.Layers(47).Classes);
%channels = [114 293 341 484 563 950];
%classes(channels)
%layer = 46;
%for layer = 3:numel(myNet.Layers)-1
arr_layer = [numel(myNet.Layers)-1];
%for layer = 4:numel(arr_layer);
%channels = myNet.Layers(arr_layer(layer), 1).NumFilters;
channels = numel(myNet.Layers(47).Classes);
%channels = 1;

layer=1
i=1
j=1
k=1
for layer=1:numel(arr_layer)
    layerName = myNet.Layers(arr_layer(layer)).Name;

    for i=1:numel(arr_pyram)
        pyram = arr_pyram(i);

        for j=1:numel(arr_iter)
            iter = arr_iter(j);
            for k=1:numel(arr_scale)
                scale = arr_scale(k);
                 curImages = deepDreamImage(myNet,arr_layer(layer),[1:channels],'Verbose',true,'PyramidLevels',pyram,...
                'NumIterations',iter,'PyramidScale',scale,'ExecutionEnvironment','gpu');
            
                curImages_tile = imtile(curImages);

                currTitle = ['model ', model_Rev, 'layer ', num2str(arr_layer(layer)), '-',layerName,' channels ',num2str(channels),' pyram ',num2str(pyram),' iter ',num2str(iter),...
                    ' scale ',num2str(scale)];
                %suptitle(currTitle);
                imwrite(curImages_tile, [output_dir, '/', currTitle,'.png']);
    %    end
    %     print(gcf,'-dpng','-r600',['./', currTitle,'.png']);
        %imwrite(currRGB,['./dreamoutput/',currTitle,'.png']);
        %close('all');
            end
        end
    end
end