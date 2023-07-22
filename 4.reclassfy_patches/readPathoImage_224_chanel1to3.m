% 패치이미지를 AI가 분석 가능한 형태로 변경
% AI 모델에 사용된 패치의 전처리와 동일하게 전처리하는 함수
% AI Input 사이즈인 224 x 224 x 3 형태로 변경
% imageFile = allPatchesFiles_org.Files{15}
function im_out = readPathoImage_224_chanel1to3(imageFile)
    finalDim = 224;
    % resize image
    im = imread(imageFile);
    if size(im,1) ~= finalDim || size(im,2) ~= finalDim
        im = imresize(im,[finalDim,finalDim]);
    end
    
    % 첫번째 호출은 배경의 얼룩을 제거하는 효과를 가져온다
    im = imadjust(rgb2gray(im), [0.0, 0.99], [0.0, 0.99], 1.0);
    
    % 두번째 호출은 각 이미지의 밝기 정도에 따라 대비를 향상하는 효과를 가져온다.
    im = imadjust(im);
    
    % AI(VGG19) input 사이즈인 224 x 224 x 3 형태로 변경
    im_out(:,:,1) = im;
    im_out(:,:,2) = im;
    im_out(:,:,3) = im;
end
