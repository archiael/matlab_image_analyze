%% blockproc에서 activateMyNet에 함수로 사용하여 이미지를 AI에 input으로 넣어 output인 활성화 벡터를 반환한다.
% input 파라미터 설명
% myNet = 학습된 CNN AI(VGG19)
% row = 4201;
% col = 5601;
% curFetch = curImage(row:(row+223), col:(col+223), :); %224 x 224 크기의 흑백처리한 이미지
% layerNum = AI에서 활성화 값을 출력할 레이어 인덱스
% outputLayerName = 분류 AI의 마지막 레이어의 이름
% rotInv
% eq_environ = 'gpu'
% sample_name = WSI 샘플 이름만 추출
% output_rgb_patch = 패치 이미지를 저장할 경로;
% locationIdx = [1001 1001]
% write_fetchImg = true
% tissuenames = myNet.Layers(end).Classes;

% output
% vec_ai : AI의 특정 레이어에서 output된 활성화값

% RGB dotting 시 blockproc에서 호출하는 
% imshow(curFetchImage_org)
% imshow(curFetch)
function vec_ai = imageNet_grayScaling_1to3(myNet, curFetch, layerNum, outputLayerName, rotInv, eq_environ, sample_name, output_rgb_patch, locationIdx, write_fetchImg, tissuenames)
    curFetchImage_org = curFetch;
    
    % 패치를 원본으로 뽑아낼 경우 전처리 안한 color 이미지가 들어오므로 AI가 분류하기위해 전처리를 패치단위로 처리
    if write_fetchImg
        curFetch = imgaussfilt(rgb2gray(curFetch), 2);
    end
    % 첫번째 호출은 배경의 얼룩을 제거하는 효과를 가져온다
    curFetch = imadjust(curFetch, [0.0, 0.99], [0.0, 0.99], 1.0);
    % 두번째 호출은 각 이미지의 밝기 정도에 따라 대비를 향상하는 효과를 가져온다.
    curFetch = imadjust(curFetch); 

    curFetchImage(:,:,1) = curFetch;
    curFetchImage(:,:,2) = curFetch;
    curFetchImage(:,:,3) = curFetch;
    
    if rotInv
        % feature vector in all rotations
        % define rotation functions
        rot180 = @(I) rot90(rot90(I));
        rot270 = @(I) rot90(rot90(rot90(I)));
        fvec1 = double(activations(myNet,      curFetchImage,outputLayerName,'ExecutionEnvironment',eq_environ));
        fvec2 = double(activations(myNet,rot90(curFetchImage),outputLayerName,'ExecutionEnvironment',eq_environ));
        fvec3 = double(activations(myNet,rot180(curFetchImage),outputLayerName,'ExecutionEnvironment',eq_environ));
        fvec4 = double(activations(myNet,rot270(curFetchImage),outputLayerName,'ExecutionEnvironment',eq_environ));

        % construct output from mean of rotated images,  reshape feature vector
        vec_ai = reshape(mean([fvec1;fvec2;fvec3;fvec4]),[1,1,myNet.Layers(layerNum).OutputSize]); 
    else 
        fvec = double(activations(myNet,curFetchImage,outputLayerName,'ExecutionEnvironment',eq_environ));
        vec_ai = reshape(fvec,[1,1,myNet.Layers(layerNum).OutputSize]);
        
        % write_fetchImg: true면 패치이미지를 output_rgb_patch에 추출한 location정보를
        % sample_name에 추가하고 AI가 분류한 최고확률 클래스명과 확률값을 붙여서 tif형태로 저장한다.
        if (write_fetchImg)
            
            % 검은색 테두리 이미지면 바로 리턴
            black_edge_rgb = (curFetchImage(1,1) == 0) + (curFetchImage(1,224) == 0) + (curFetchImage(224,1) == 0) + (curFetchImage(224,224) == 0);
            if(black_edge_rgb > 0)
                return;
            end

            % all white == 224*224*3 = 150528
            % half white > 25000
            % white pixel(255,255,255)인 1x1x3 벡터값이 3만개 이상이면 리턴
            whitemat = zeros(224,224)+255;
            if(sum(sum(curFetchImage==whitemat)) > 30000)
              return;
            end
            
            % 활성화 확률 벡터값중 가장 큰 값을 가진 class로 정렬
            sort_1_9 = sort(vec_ai(1,1,:),'descend'); % descend : 내림차순 정렬
            % 클래스 인덱스 배열 생성
%             class_num = 1:numel(tissuenames);
            
            % 최대값을 가진 클래스 확률값이 몇번째인지 체크
            bClassIdx = vec_ai(1,1,:) == sort_1_9(1);
            
            sort1_class = string(tissuenames(bClassIdx));
            % temp_class_num : 특정 클래스의 패치만 저장하고 싶을 때 사용
            % temp_class_num = class_num(bClassIdx);

            % 이미지를 각 분류한 폴더별로 저장한다.
            % blockproc.m 460 line에서 function serialLoop의 block_struct만 fun에 넘겨주던것을
            % row와 col 정보를 넘겨줘서 자를때 어디서 잘랐는지 정보를 이름에 부여
            target_path = strcat(output_rgb_patch,'/',sort1_class(1),'/', sample_name ...
            ,'_row_',num2str(locationIdx(1)),'_col_',num2str(locationIdx(2)) ...
            ,'-', sort1_class,'_', num2str(round(sort_1_9(1),1)) ...
            ,'.tif');
            imwrite(curFetchImage_org, target_path);
        end
    end
end