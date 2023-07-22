%% 2022-06-08 : 15개 클래스로 패치를 분류한 후 학습에 사용
%% 학습을 위한 사전 세팅 정보
% training_inputPath : 폴더에 패치로 학습
% validation_inputPath : 폴더에 패치로 학습 도중에 정확도 검증
% save_outputPath : 폴더에 학습&테스트한 모델 결과파일 저장
% tissuenames : 학습할 클래스 목록, 학습셋의 폴더명으로 자동 클래스 지정된 목록과 동일한지 검증용
% colors : 클래스별 이미지 변환을 위한 RGB 세팅값
%% 학습 완료된 AI 검증을 위한 사전 세팅 정보

% arr_testing_inputPath 에 지정한 '폴더들'의 패치로 각각 학습 완료 후 정확도 검증


% 작업 공간 초기화
clear all, close all; % 모든 변수 지우기, 모든 Plot 창 닫기
format compact; % 출력 표시 형식 Style을 출력값이 더 많이 표시되도록 빈라인을 억제합니다.
clc; % 명령창 지우기

% AI 학습은 2번으로만
% 이미지분석은 1번으로도 가능
gpuDevice(2);

% subroutines 함수파일이 있는 폴더를 path에 추가
% @readPathoImage_224_gray_blur_chanel1to3를 못찾는다는 에러가 나면
% 아래 addpath()를 사용하여 readPathoImage_224_gray_blur_chanel1to3.m
% 파일이 있는 경로를 아래 [] 대괄호 사이에 문자열로 넣어준다.
% addpath("");

%% AI 학습 사전 세팅 시작
% 멀티 GPU 사용 시
eq_environ = 'parallel';

% 모델명을 지정하면 해당 모델명 지정
model_Rev = 'Rev0.01.13_LEICA_C04C05';

% 학습용 이미지 폴더 지정
training_inputPath =  'C:\Y330_FINAL\old_data\TRAINING_ver05_preprocessing';

% 검증셋 이미지 폴더 지정
validation_inputPath =  'C:\Y330_FINAL\TUNING_ver05';

% 학습한 AI 모델파일이 저장될 경로
save_outputPath = ['C:\train_VGG19\2.trained_model\Rev0.01.xx_history\', model_Rev];
mkdir(save_outputPath); 

% 학습할 AI 모델의 클래스명을 지정
% training_set의 레이블과 비교하여 다른지 체크하기 위함
% 15클래스
tissuenames = { ...
 'C-01', ...
 'C-02', ...
 'C-03', ...
 'C-04', ...
 'C-05', ...
 'C-06', ...
 'C-07', ...
 'C-08', ...
 'C-09', ...
 'C-10', ...
 'C-11', ...
 'C-12', ...
 'C-13', ...
 'C-14', ...
 'C-15'
};

% 클래스 순서에 맞게 색상 지정
% 색상 : RGB dotting시 확률값을 변환할 색상(Red, Green Blue)
colors      = [ ...
 192,40,0; ...
 122,26,0; ...
 91,21,21; ...
 246,108,4; ...
 255,142,91; ...
 0,130,220; ...
250,220,110; ...
 130,190,20; ...
 72,173,36; ...
 52,143,30; ...
200,200,200; ...
 0,43,72; ...
248,116,229; ...
 64,64,64; ...
 255 255 255;]/255;

% AI 학습 사전 세팅 종료

%% AI 학습용 이미지 로딩

disp('학습셋 불러오기 시작');
% training_inputPath경로에 하위폴더를 포함한 이미지파일들을 하위 조직분류 폴더이름을 레이블을 지정하면서 imageDatastore 객체를 만듭니다.
training_set = imageDatastore(training_inputPath,'IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.tif'});

% 이미지 실제 로딩 시 전처리 함수
% 그레이스켈링 & 블러 처리된 패치를 1채널 -> 3채널로 변경
training_set.ReadFcn = @readPathoImage_224_chanel1to3; % read and resize images to 224 by subroutines/readPathoImage_224.m

training_tbl = countEachLabel(training_set); %# countEachLabel() : 폴더별(레이블) 파일 개수를 읽어들인다
disp(training_tbl);
training_categories = training_tbl.Label; % training_categories 변수에 카테고리(폴더이름)를 저장합니다.
class_count = numel(unique(training_set.Labels));
disp(strcat("학습할 클래스 카운트 갯수 ", num2str(class_count)));
disp('학습셋 불러오기 종료');

% 학습용 패치 이미지 클래스별 분포를 히스토그램으로 표시
histogram(training_set.Labels)
set(gca,'TickLabelInterpreter','none')

% 불러온 이미지셋의 랜덤 이미지 미리보기
numObs = length(training_set.Labels);
numObsToShow = numel(training_categories);
idx = randperm(numObs,numObsToShow);
imshow(imtile(training_set.Files(idx),'GridSize',[3 5],'ThumbnailSize',[224 224]))

% 지정한 학습 레이블 개수와 불러온 학습세트의 레이블 개수가 일치하는지 체크
if ~(numel(unique(training_set.Labels)) == numel(tissuenames))
    disp("학습셋에 레이블 숫자가 학습하려는 레이블 숫자와 일치하지 않습니다. tissuenames와 training_set.Labels 확인바랍니다.!!!!!!!!!!!!!!!!!!!");
    return;
end


%% AI 검증셋 불러오기
disp('검증셋 불러오기 시작');
% validation_inputPath경로에 하위폴더를 포함한 이미지파일들을 하위 조직분류 폴더이름을 레이블을 지정하면서 imageDatastore 객체를 만듭니다.
validation_set = imageDatastore(validation_inputPath,'IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.tif'});

% 컬러 패치 이미지 로딩 시 전처리 함수
% 패치 이미지 전처리 후 1채널 이미지를 3채널로 변경
validation_set.ReadFcn = @readPathoImage_224_gray_blur_chanel1to3; % read and resize images to 224 by subroutines/readPathoImage_224.m

validation_tbl = countEachLabel(validation_set); %# countEachLabel() : 폴더별(레이블) 파일 개수를 읽어들인다
validation_categories = validation_tbl.Label; % validation_categories 변수에 카테고리(폴더이름)를 저장합니다.
class_count = numel(unique(validation_set.Labels));
disp(strcat("검증할 클래스 카운트 갯수 ", num2str(class_count)));
disp('검증셋 불러오기 종료');

%% AI 모델 : AI 모델 (VGG19) 사전학습 파일 불러오기 (해당하는 모델 주석 풀기)
rawnet = vgg19();

% 모델 레이어의 마지막으로부터 3번째(end-2) 계층과 마지막(end) 계층을 초기화
layers = rawnet.Layers;
% training_set 객체에 유니크한 레이블(폴더이름) 개수로 최종 분류의 개수를 설정
layers(end-2) = fullyConnectedLayer(numel(tissuenames), 'name', 'predictions');

% 네트워크 아키텍처에서 최종 output 레이어에 분류계층을 생성함
layers(end) = classificationLayer('name', 'ClassificationLayer_predictions');

% imageInputSize 변수에 첫번째 레이어의 Image Input 사이즈(224x224로 설정)를 가져온다.
imageInputSize = layers(1).InputSize(1:2);
disp(['sucessfully loaded&modified network, input size is ', num2str(imageInputSize)]);

% analyzeNetwork(layers) % AI 모델 매트랩 시각화
% AI 모델 세팅 끝

%% 학습데이터 증대
% 이미지 데이터 증대를 위한 사전처리 옵션 지정
% 네트워크가 과적합되는 것을 방지하고 훈련 이미지의 정확한 세부 정보가 기억되지 않도록 하는데 도움이 됨
% 세로, 가로 축을 따라 훈련 이미지를 무작위로 뒤집는 옵션 추가
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true);
augmented_training_set = augmentedImageDatastore(imageInputSize,training_set,'DataAugmentation',imageAugmenter);

%% TRAIN
% 딥러닝 훈련: 검증 정확도를 위한 옵션
% 'sgdm'- SGDM(Stochastic Gradient Descent with Momentum: 모멘텀을 사용한 확률적 경사하강법) 최적화 함수를 사용합니다
%  논문에서는 이 방법과 아래 2가지 방법에 대한 비교수치를 제시하지 않음
% or 'rmsprop'? RMSProp 최적화 함수를 사용합니다. 'SquaredGradientDecayFactor' 이름-값 쌍 인수를 사용하여 제곱 기울기 이동평균의 감쇠율을 지정할 수 있습니다.
% or 'adam'? Adam 최적화 함수를 사용합니다. 'GradientDecayFactor' 이름-값 쌍 인수를 사용하여 기울기 이동평균의 감쇠율을, 'SquaredGradientDecayFactor' 이름-값 쌍 인수를 사용하여 제곱 기울기 이동평균의 감쇠율을 지정할 수 있습니다.
% MiniBatchSize- 각 반복마다 360개의 관측값을 갖는 미니배치를 사용 초기값은 360
% MaxEpochs- 최대 Epoch 횟수를 8로 설정 초기값은 8
% InitialLearnRate- 훈련에 사용할 초기학습률(초기값 sgdm(0.01), rmsprop(0.001), adam(0.001))
% VerboseFrequency- 상세 출력의 빈도
% ValidationData : 훈련 중에 검증에 사용할 데이터로, 검증 예측 변수와 검증 응답 변수를 포함하는 데이터저장소, 테이블 또는 셀형 배열로 지정됩니다.
% Shuffle : 검증 데이터는 'Shuffle' 값에 따라 섞입니다. 'Shuffle' 값이 'every-epoch'인 경우, 검증 데이터는 각 신경망 검증 전에 섞입니다.
% 훈련진행 상황을 모니터링 하고 싶어 'Plots','training-progress' 옵션을 추가
% ExecutionEnvironment gpu로 하여 GPU를 사용하여 학습 (기타 옵션 기본값 'auto', 'cpu', 'multi-gpu', 'parallel')
opts = trainingOptions('sgdm',...
    'MiniBatchSize',128,...           
    'MaxEpochs',8,...               
    'InitialLearnRate',3e-4,...       
    'VerboseFrequency',1,...
    'ValidationData', validation_set, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ExecutionEnvironment', eq_environ);

% 학습 옵션 출력
disp(opts)


% 이전버전 매트랩 GPU 가벼운 연산걸어서 시동거는 코드 시작
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
nnet.internal.cnngpu.reluForward(1);
ile=gpuArray(0.0001);

myNet = trainNetwork(augmented_training_set, layers, opts);
disp('successfully training images');

% 학습한 AI 중간 저장
save(strcat(save_outputPath,'/', model_Rev, '.mat'),'myNet', 'training_categories', 'training_tbl', 'colors', 'model_Rev');






%% 학습완료된 모델의 테스트
% 테스트셋을 이용하여 모델의 성능을 구하고
% 테스트셋을 예측한 폴더별로 재분류하여 학습이 잘되었는지 확인하실 수 있습니다.

% arr_testing_name : 테스트셋의 명칭
arr_testing_name = ["tuning_C04C05","test_C04C05"];
% 테스트셋 전체경로
arr_testing_inputPath = ["C:\Y330_FINAL\TUNING_ver05", "C:\Y330_FINAL\TEST_ver05"];

% test_i = 1;
for test_i = 1:numel(arr_testing_inputPath)
    % 이제 훈련된 모델로 실제 분류할 이미지를 로딩, ImageDatastore 객체를 생성함
    disp('loading TESTING images');
    testing_set = imageDatastore(arr_testing_inputPath(test_i),'IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.tif'});
    
    % 이미지 실제 로딩 시 전처리 함수
    % 그레이스케일링 + blur 처리
    testing_set.ReadFcn = @readPathoImage_224_gray_blur_chanel1to3; % read and resize images to 224 by subroutines/readPathoImage_224.m
    
    testing_categories = testing_tbl.Label;
    numel(unique(testing_set.Labels));
    class_count = numel(unique(testing_set.Labels));
    
    % 불러온 테스트셋 정보 출력
    disp(testing_tbl);
    disp(strcat("테스트할 클래스 카운트 갯수 ", num2str(class_count)));
    disp('successfully loaded TESTING images');
    %figure, imshow(preview(testing_set));
    
    testing_tbl = countEachLabel(testing_set);
    
    % AI 모델로 테스트셋 패치를 예측시킴
    % myNet: 트레이닝된 분류모델을 사용하여 데이터 분류
    % testing_set: 실제 분류할 이미지 객체
    % predictedLabels : 예측한 클래스명
    % predictedScore : 예측한 클래스별확률값 (합은 1)
%     eq_environ = 'gpu'; % 걸린 시간 : 186초
%     eq_environ = 'multi-gpu' % 걸린 시간 : 88초
%     eq_environ = 'parallel'; % 걸린 시간 : 85초
    tic
    [predictedLabels, predictedScore] = classify(myNet, testing_set, 'MiniBatchSize', 128, 'ExecutionEnvironment', eq_environ);
    toc
    disp('successfully classify TESTING images');
    
    %% assess accuracy, show confusion matrix
    % 정확성 평가, confusion matrix 표시 단계
    labels_ground = testing_set.Labels; % 테스팅 이미지셋의 실제 폴더 라벨
    labels_pred = predictedLabels;      % 트레이닝 모델에 따라 분류된 이미지 라벨
    tissuenames = unique(myNet.Layers(end).ClassNames);
    
    tissuenames_accuracy =[{('"구분" "Sensitivity" "Precision" "Specificity" "Recall" "Accuracy"')}; tissuenames]; % 각 클래스 별 정확도를 배정
    
    Sensitivity_class = zeros(1,numel(tissuenames)); % Sensitivity(민감도*)
    Precision_class   = zeros(1,numel(tissuenames)); % Precision  (정밀도*)
    Specificity_class = zeros(1,numel(tissuenames)); % Specificity(특이도)
    Recall_class      = zeros(1,numel(tissuenames)); % Recall     (재현율)
    Accuracy_class    = zeros(1,numel(tissuenames)); % Accuracy   (정확도)
    
    % 암인 테스트셋 클래스 패치이미지들의 총합 (TUM or AB1N+AB2MM+AB3S)
    cnt_tot_TUM = 0;
    % 암이 아닌 테스트셋 클래스 패치들의 총합 (ADI ... STR)
    cnt_tot_ETC = 0;
    
    for j = 1:numel(tissuenames)
        
        logical_labels_ground   = labels_ground==tissuenames(j);
        logical_labels_Predict  = labels_pred==tissuenames(j);
        
        % TP : 실제값(Positive) + 예상 TRUE = 해당 클래스 맞춘 숫자
        TP = sum(logical_labels_ground & logical_labels_Predict);
        
        % TN : 실제값(Negative) + 예상 FALSE = 다른 클래스들 맞춘 숫자 합
        TN = sum(~logical_labels_ground & ~logical_labels_Predict);
        
        % FP : 실제값(Positive) + 예상 FALSE
        FP = sum(~logical_labels_ground & logical_labels_Predict);
        
        % FN : 실제값(Negative) + 예상 TRUE
        FN = sum(logical_labels_ground & ~logical_labels_Predict);
        
        % Sensitivity(민감도=재현율) = ( TP / TP + FN ) 양성 중 맞춘 양성의 수
        Sensitivity_class(j) = TP / (TP + FN);
    
        % Precision  (정밀도) = (TP / TP + FP) 양성이라고 판정 한 것 중에 실제 양성 수
        Precision_class(j)   = TP / (TP + FP);
    
        % Specificity(특이도) = (TN / FP + TN) 음성 중 맞춘 음성의 수
        Specificity_class(j) = TN / (FP + TN);
        
        % Recall     (재현율=민감도) = (TP / TP + FN) 전체 양성 수에서 검출 양성 수
        Recall_class(j)      = TP / (TP + FN);
    
        % Accuracy   (정확도) = (TP + TN / TP + FN + FP + TN)  전체 개수 중에서 양성과 음성을 맞춘 수
        Accuracy_class(j)       = (TP + TN) / (TP + FN + FP + TN);
        
        tissuenames_accuracy{j+1} = strcat('"', tissuenames(j), ' image Accuracy is"' ...
            , " ", num2str(round(Sensitivity_class(j), 2)) ...
            , " ", num2str(round(Precision_class(j)  , 2)) ...
            , " ", num2str(round(Specificity_class(j), 2)) ...
            , " ", num2str(round(Recall_class(j)     , 2)) ...
            , " ", num2str(round(Accuracy_class(j)   , 2)) ...
            );
    end
    
    disp(['민감도 평균 ',num2str(mean(Sensitivity_class))]);
    disp(['정밀도 평균 ',num2str(mean(Precision_class  ))]);
    disp(['특이도 평균 ',num2str(mean(Specificity_class))]);
    disp(['재현율 평균 ',num2str(mean(Recall_class     ))]);
    PerItemAccuracy = (mean(Sensitivity_class) + mean(Precision_class  )) / 2;
    disp(['민감도+정밀도 평균 ',num2str(PerItemAccuracy)]);
    
    tissuenames_accuracy(end+1) = {strcat('"per image total accuracy is "', " ", num2str(PerItemAccuracy))};
    tissuenames_accuracy(end+1) = {strcat('"Training Image Total Count "', " ", num2str(sum(training_tbl.Count(:))))};
    for k=1:numel(training_tbl.Label)
        tissuenames_accuracy(end+1) = {strcat('"Training__', string(training_tbl.Label(k)),'"', " ", string(training_tbl.Count(k)))};
    end
    
    tissuenames_accuracy(end+1) = {strcat('"Test Image Total Count "', " ", num2str(sum(testing_tbl.Count(:))))};
    for k=1:numel(testing_tbl.Label)
        tissuenames_accuracy(end+1) = {strcat('"Testing__', string(testing_tbl.Label(k)), '"', " ", string(testing_tbl.Count(k)))};
    end
    
    write(cell2table(tissuenames_accuracy), strcat(save_outputPath,'/', arr_testing_name(test_i),'_sub_',model_Rev, 'class', num2str(class_count), '_accuracy_', num2str(PerItemAccuracy), '.txt'));
    
    allgroups = cellstr(unique(labels_ground));             % cellstr(A):  A를 문자형 벡터로 구성된 셀형 배열로 변환
    figure('Position', [0 0 1200 700]);
    cm = confusionchart(labels_ground,labels_pred);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    cm.Title = strcat(strrep(model_Rev, '_', '-'), ' Confusion Matrix');
    print(gcf,'-dpng','-r600',strcat(save_outputPath,'/', arr_testing_name(test_i), 'sub_test_heatmap_Rev_', model_Rev,'_accuracy_', num2str(PerItemAccuracy),'.png'));
    
    save(strcat(save_outputPath,'/', arr_testing_name(test_i),'_class_', ...
        num2str(class_count),'_', model_Rev, '.mat'),...
        'myNet', 'colors', 'model_Rev', 'PerItemAccuracy', 'training_categories', 'training_tbl', 'testing_tbl', 'predictedLabels', 'predictedScore');
    
    %% AI 학습한 모델을 이용하여 테스트셋의 패치 이미지를 재분류(복사) 한다.
    output_path_test = strcat(arr_testing_inputPath(test_i), '_', model_Rev, '_Reclassify');
    mkdir(output_path_test);
    winopen(output_path_test);
    
    for j = 1:numel(tissuenames)
        for k = 1:numel(tissuenames)
            dir = strcat(output_path_test,'/',string(tissuenames(j)),'_to_',string(tissuenames(k)));
            mkdir(dir);
        end
    end
    
    allPatchesFiles_Files = string(testing_set.Files);
    parfor i=1:numel(allPatchesFiles_Files)
        % 사람이 분류한 라벨과 예측라벨이 같은지 체크하기 위해 이미지파일들을
        % 원본라벨_to_예측 폴더로 복사한다.
        target = strcat(output_path_test, '/', string(labels_ground(i)),'_to_', string(labels_pred(i)));
        copyfile(string(allPatchesFiles_Files(i)), target);
    end
end