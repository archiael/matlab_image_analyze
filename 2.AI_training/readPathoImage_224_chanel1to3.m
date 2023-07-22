% ImageDatastore에 ReadFcn에 적용하여 
% AI 패치 학습용으로 입력된 이미지세트(grayscale, 가우스블러, 밝기, 대비 증대)를 
% AI Input 사이즈인 224 x 224 x 3 형태로 im_out 변수로 내보낸다.
% input : imageDatastore에 Files 필드에 저장된 파일 전체 경로
function [ im_out ] = readPathoImage_224_chanel1to3(imageFilePath)
    finalDim = 224;
    
    % read image
    im = imread(imageFilePath);

    % resize image
    if size(im,1) ~= finalDim || size(im,2) ~= finalDim
        im = imresize(im,[finalDim,finalDim]);
    end
    
    % 1채널 -> 3채널로 변경
    im_out(:,:,1) = im;
    im_out(:,:,2) = im;
    im_out(:,:,3) = im;
end