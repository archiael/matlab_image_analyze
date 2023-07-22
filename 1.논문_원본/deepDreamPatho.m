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
% 마지막 레이어의 학습된 특징을 시각화

% 'PyramidLevels' ? 피라미드 단계의 개수
% 출력 영상을 생성하는 데 사용된 다중 해상도 영상 피라미드 단계의 개수로, 양의 정수로 지정됩니다. 
% 연산량을 늘리는 대신 더 큰 출력 영상을 생성하려면 피라미드 단계의 개수를 늘리십시오. 
% 초기 영상과 크기가 같은 영상을 생성하려면 단계의 개수를 1로 설정하십시오.
% 2021-03-10 : 5로 설정하는게 가장 적당한 퀄리티의 특징을 표현해줌을 확인
arr_pyram = [5]
% 'NumIterations' ? 피라미드 단계당 반복 횟수
% 10 (디폴트 값) | 양의 정수
% 피라미드 단계당 반복 횟수로, 양의 정수로 지정됩니다. 연산량을 늘리는 대신 더 상세한 영상을 생성하려면 반복 횟수를 늘리십시오.
% 2021-03-09 : 75가 가장 적당한 퀄리티의 특징을 표현해줌을 확인
arr_iter = [75]
% 'PyramidScale' ? 피라미드 단계 간 스케일
%1.4 (디폴트 값) | 1보다 큰 스칼라 값
%각 피라미드 단계 간의 스케일로, 1보다 큰 스칼라 값으로 지정됩니다. 
% 출력 영상에 미세한 세부 사항을 반용하려면 피라미드 스케일을 줄이십시오. 
% 피라미드 스케일을 조정하면 신경망 시작 부분에 있는 계층에서 정보가 많은 영상을 생성하는 데 도움이 됩니다.
% 2021-03-10 : 1.1이 세부적인 특징까지 잡는것으로 보여짐
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