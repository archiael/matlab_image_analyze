%% blockproc���� activateMyNet�� �Լ��� ����Ͽ� �̹����� AI�� input���� �־� output�� Ȱ��ȭ ���͸� ��ȯ�Ѵ�.
% input �Ķ���� ����
% myNet = �н��� CNN AI(VGG19)
% row = 4201;
% col = 5601;
% curFetch = curImage(row:(row+223), col:(col+223), :); %224 x 224 ũ���� ���ó���� �̹���
% layerNum = AI���� Ȱ��ȭ ���� ����� ���̾� �ε���
% outputLayerName = �з� AI�� ������ ���̾��� �̸�
% rotInv
% eq_environ = 'gpu'
% sample_name = WSI ���� �̸��� ����
% output_rgb_patch = ��ġ �̹����� ������ ���;
% locationIdx = [1001 1001]
% write_fetchImg = true
% tissuenames = myNet.Layers(end).Classes;

% output
% vec_ai : AI�� Ư�� ���̾�� output�� Ȱ��ȭ��

% RGB dotting �� blockproc���� ȣ���ϴ� 
% imshow(curFetchImage_org)
% imshow(curFetch)
function vec_ai = imageNet_grayScaling_1to3(myNet, curFetch, layerNum, outputLayerName, rotInv, eq_environ, sample_name, output_rgb_patch, locationIdx, write_fetchImg, tissuenames)
    curFetchImage_org = curFetch;
    
    % ��ġ�� �������� �̾Ƴ� ��� ��ó�� ���� color �̹����� �����Ƿ� AI�� �з��ϱ����� ��ó���� ��ġ������ ó��
    if write_fetchImg
        curFetch = imgaussfilt(rgb2gray(curFetch), 2);
    end
    % ù��° ȣ���� ����� ����� �����ϴ� ȿ���� �����´�
    curFetch = imadjust(curFetch, [0.0, 0.99], [0.0, 0.99], 1.0);
    % �ι�° ȣ���� �� �̹����� ��� ������ ���� ��� ����ϴ� ȿ���� �����´�.
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
        
        % write_fetchImg: true�� ��ġ�̹����� output_rgb_patch�� ������ location������
        % sample_name�� �߰��ϰ� AI�� �з��� �ְ�Ȯ�� Ŭ������� Ȯ������ �ٿ��� tif���·� �����Ѵ�.
        if (write_fetchImg)
            
            % ������ �׵θ� �̹����� �ٷ� ����
            black_edge_rgb = (curFetchImage(1,1) == 0) + (curFetchImage(1,224) == 0) + (curFetchImage(224,1) == 0) + (curFetchImage(224,224) == 0);
            if(black_edge_rgb > 0)
                return;
            end

            % all white == 224*224*3 = 150528
            % half white > 25000
            % white pixel(255,255,255)�� 1x1x3 ���Ͱ��� 3���� �̻��̸� ����
            whitemat = zeros(224,224)+255;
            if(sum(sum(curFetchImage==whitemat)) > 30000)
              return;
            end
            
            % Ȱ��ȭ Ȯ�� ���Ͱ��� ���� ū ���� ���� class�� ����
            sort_1_9 = sort(vec_ai(1,1,:),'descend'); % descend : �������� ����
            % Ŭ���� �ε��� �迭 ����
%             class_num = 1:numel(tissuenames);
            
            % �ִ밪�� ���� Ŭ���� Ȯ������ ���°���� üũ
            bClassIdx = vec_ai(1,1,:) == sort_1_9(1);
            
            sort1_class = string(tissuenames(bClassIdx));
            % temp_class_num : Ư�� Ŭ������ ��ġ�� �����ϰ� ���� �� ���
            % temp_class_num = class_num(bClassIdx);

            % �̹����� �� �з��� �������� �����Ѵ�.
            % blockproc.m 460 line���� function serialLoop�� block_struct�� fun�� �Ѱ��ִ�����
            % row�� col ������ �Ѱ��༭ �ڸ��� ��� �߶����� ������ �̸��� �ο�
            target_path = strcat(output_rgb_patch,'/',sort1_class(1),'/', sample_name ...
            ,'_row_',num2str(locationIdx(1)),'_col_',num2str(locationIdx(2)) ...
            ,'-', sort1_class,'_', num2str(round(sort_1_9(1),1)) ...
            ,'.tif');
            imwrite(curFetchImage_org, target_path);
        end
    end
end