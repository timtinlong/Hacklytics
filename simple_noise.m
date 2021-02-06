root_path = 'D:\hacklytics\chest_xray\';
test_path = fullfile(root_path, 'test');
save_path = fullfile(root_path, 'test_noise');

folders = dir(test_path);

create_folder = fullfile(save_path);
if ~exist(create_folder, 'dir')
   mkdir(create_folder)
end
create_folder = fullfile(save_path,'NORMAL');
if ~exist(create_folder, 'dir')
   mkdir(create_folder)
end
create_folder = fullfile(save_path,'PNEUMONIA');
if ~exist(create_folder, 'dir')
   mkdir(create_folder)
end


for i = 3:size(folders,1)
    full_path = fullfile(test_path, folders(i).name);
    images_lst = dir(full_path);
    for ii = 3:size(images_lst,1)
        curr_im = imread(fullfile(full_path,images_lst(ii).name));
%         figure
%         subplot(1,2,1)
%         imshow(curr_im)
        curr_im = imresize(curr_im, [256 256], 'bicubic');
        curr_im = imnoise(curr_im ,'gaussian',0,0.01);
        
        % https://www.mathworks.com/matlabcentral/answers/442002-image-transformation-histogram-shifting
        curr_im = curr_im + 25; % histogram shift to the right
        % setting the value range between 25 to 225;
        curr_im(curr_im > 225) = 225;
        curr_im(curr_im < 25) = 25;
        imwrite(curr_im, fullfile(save_path, folders(i).name, images_lst(ii).name))
%         subplot(1,2,2)
%         imshow(curr_im)
        
        
    end
   
end