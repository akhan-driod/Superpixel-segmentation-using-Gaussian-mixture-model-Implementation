% LoadApplesScript.m
% script to load the apple images and masks into MATLAB cells

if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') )
    display('Please change current directory to the parent folder of both apples/ and testApples/');
end

% Note that cells are accessed using curly-brackets {} instead of parentheses ().
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';

I_test_apples = cell(5,1);
I_test_apples{1} = 'testApples/Apples_by_MSR_MikeRyan_flickr.jpg';
I_test_apples{2} = 'testApples/audioworm-QKUJj2wmxuI-original.jpg';
I_test_apples{3} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg';
I_test_apples{4} = 'testApples/internet_1.jpg';
I_test_apples{5} = 'testApples/internet_2.jpg';


%training data
apples = cell(3,1);
applesMasks = cell(3,1);
for iImage = 1:3,
    
    %image
    im = double(imread(  Iapples{iImage}   )) / 255; %dims: (width,height,3)
    apples{iImage} = im;
    
    %image mask
    mask = imread(  IapplesMasks{iImage}   );
    applesMasks{iImage} = mask;
    
end;
  
%testing data
test_apples = cell(5,1);                                                                                                                                                                                            
for iImage = 1:5,
    im_test = double(imread(  I_test_apples{iImage}   )) / 255; %dims: (width,height,3)
    test_apples{iImage} = im_test;
    
end;
