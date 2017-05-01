clc;
clear all;
close all;
% warning off

pkg load image
pkg load communications

%% *------------------- Picode Encoding------------------------------*
%% Input image

% [fn,pn]=uigetfile({'*.git;*.png;*.jpg','Select the Cover Image'});
% if fn==0
    % return
% end

% im=imread( [pn fn] );
im = imread('lena.jpg');

figure
imshow(im);
title('Cover image');

% imwrite(im,'pep.png');

new=im;

%% Image Resizal
im=imresize (im,[116 116]);  % for 29 x 29 modules
figure;
imshow(im);
title('Resized cover Image');

%% RGB to Gray
g=rgb2gray(im);
% figure;
% imshow(g);
% title('Cover image in gray');


bitstream=randi([0 1],1,1160);
l=length(bitstream);
B=bitstream;
% %% Channel EnCoding using Hamming
m = 8;           % Number of bits per symbol or letter
n = 2^m - 1;     % n=Codeword length,255;
k=n-m;
%
B = encode(bitstream,n,k,'hamming');

%% Block Division: Module size=4x4;

[r,c]=size(g);  % Y component and gray component is almost same. here we use gray image 'g'.

Rsize=4;
CSize=4;

Rows=floor(r/Rsize);
Col=floor(c/CSize);

R=[Rsize*ones(1,Rows)];
C=[CSize*ones(1,Col)];

I=mat2cell(g,R,C);

[s1,s2]=size(I);

%% calculation of ei (inner) and eo (outer)

for i=1:s1
    for j=1:s2
         for r=1:4
             for c=1:4
                 if (r==2||r==3||c==2||c==3) % Inner Pixel
                     ei(i,j) = mean2(I{i,j}(r,c));
                 end

                 if (r==1||c==4||r==4||c==1) % Outer Pixel
                     eo(i,j) = mean2(I{i,j}(r,c));
                 end

             end
         end
    end
end

%% Contrast Evaluation of each block in I

% Input Image=I
mu_bright = zeros(s1,s2); % initialisation is important.else produces an error.
mu_dark = zeros(s1,s2);

for i=1:s1
    for j=1:s2

        maxi=max(max(I{i,j}));
        mini=min(min(I{i,j}));

        thresh=(maxi+mini)/2;
        thresh=round(thresh);

        % To access elements inside a cell,
        for sub_row=1:4
            for sub_col=1:4

                % Mean of Bright pixels
                if I{i,j}(sub_row,sub_col)> thresh    % pixels from thresh to rest of elements are assigned as foregroun
                     mu_bright(i,j)=mean2(I{i,j}(sub_row,sub_col));
                end

                % Mean of Dark Pixels
                if I{i,j}(sub_row, sub_col)<=thresh     % pixels from 1 to thresh are assigned as background
                      mu_dark(i,j)=mean2(I{i,j}(sub_row, sub_col));
                end

                C(i,j)=mu_bright(i,j)-mu_dark(i,j);

            end
        end
   end
end

%% Equation Parameters

lambda=25; % Quality Parameter
eta=0.1;
R=3;       % Ratio b/w modulation intensity values of outer and inner parts.
% ei       % avg clipped intensity values for inner pixels
% eo       % avg clipped intensity values for outer pixels

%% Delta_I: Computation of Adaptive Intensity Parameter in I

delta_I=cell(s1,s2);

for i=1:s1
    for j=1:s2
        for r=1:4
            for c=1:4
                if (r==2||r==3||c==2||c==3) % Inner pixel
                    delta_I{i,j}(r,c) = ceil((C(i,j)+eta)/lambda).*lambda + ei(i,j);
                end

                if (r==1||c==4||r==4||c==1) % Outer Pixel
                    delta_I{i,j}(r,c) = ceil((C(i,j)+eta)/lambda).*lambda + eo(i,j);
                end

            end
        end
    end
end

%% Bit Padding with zeros: since the blocks are of size 29*29=841, each bit has to be given to each block.
% size has to be made of same size;

%B(end+1:841)=0; % pads zeros from next
%C1=B(1:841);
%V1=vec2mat(C1,65);
C1=B;

%% Second Term in Modulation

term=cell(s1,s2);

count=0;

for i=1:s1
    for j=1:s2

        count=count+1;

        for r= 1:4
            for c= 1:4

                if (r==2||r==3||c==2||c==3) % Inner
                    term{i,j}(r,c) = ((-1)^C1(count)).*(delta_I{i,j}(r,c));
                end

                if (r==1||c==4||r==4||c==1) % Outer
                     term{i,j}(r,c) = ((-1)^C1(count)).*((delta_I{i,j}(r,c))/R);
                end

            end
        end

    end

end

%% Modulation

Im=cell(s1,s2);

for i=1:s1
    for j=1:s2
        for r= 1:4
            for c= 1:4

                if (r==2||r==3||c==2||c==3)
                      Im{i,j}(r,c) = I{i,j}(r,c) - term{i,j}(r,c);
                end

                if (r==1||c==4||r==4||c==1)
                      Im{i,j}(r,c) = I{i,j}(r,c)+ term{i,j}(r,c);
                end

            end
        end
   end
end

I_modulated=cell2mat(Im);
% figure;
% imshow(I_modulated);
% title('Modulated Image In Gray Form');

Im_new=Im;

%% Mapping

% Bit 1  mapping : if inner = 255, outer to 0.
for i=1:s1
    for j=1:s2
        for r=1:4
            for c=1:4
                if (r==2||r==3||c==2||c==3)

                    if Im{i,j}(r,c)>1; % inner non zero pixels

                        Im{i,j}(r,c)=255;

                        % outer pixels

                        % first row
                        Im{i,j}(1,1)=0;
                        Im{i,j}(1,2)=0;
                        Im{i,j}(1,3)=0;
                        Im{i,j}(1,4)=0;

                        % last row
                        Im{i,j}(4,1)=0;
                        Im{i,j}(4,2)=0;
                        Im{i,j}(4,3)=0;
                        Im{i,j}(4,4)=0;

                        % first column
                        Im{i,j}(1,1)=0;
                        Im{i,j}(2,1)=0;
                        Im{i,j}(3,1)=0;
                        Im{i,j}(4,1)=0;

                        % last column
                        Im{i,j}(1,4)=0;
                        Im{i,j}(2,4)=0;
                        Im{i,j}(3,4)=0;
                        Im{i,j}(4,4)=0;

                    end
                end
            end
        end
    end
end

% Bit 0 Mapping
for i=1:s1
    for j=1:s2
        for r=1:4
            for c=1:4
                if (r==2||r==3||c==2||c==3)

                    if Im{i,j}(1:4,1:4)==0;

                        % outer Pixels

                        % first row
                        Im{i,j}(1,1)=255;
                        Im{i,j}(1,2)=255;
                        Im{i,j}(1,3)=255;
                        Im{i,j}(1,4)=255;

                        % last row
                        Im{i,j}(4,1)=255;
                        Im{i,j}(4,2)=255;
                        Im{i,j}(4,3)=255;
                        Im{i,j}(4,4)=255;

                        % first column
                        Im{i,j}(1,1)=255;
                        Im{i,j}(2,1)=255;
                        Im{i,j}(3,1)=255;
                        Im{i,j}(4,1)=255;

                        % last column
                        Im{i,j}(1,4)=255;
                        Im{i,j}(2,4)=255;
                        Im{i,j}(3,4)=255;
                        Im{i,j}(4,4)=255;
                    end

                end

            end

        end

    end
end

I_modulated=cell2mat(Im);
% figure;
% imshow(I_modulated);
% title('Modulated Image In Gray Form');


%% Conversion of I_mod to colour channels

for k=1:3
    for i=1:116
        for j=1:116
               I_modulated(i,j,k) = I_modulated(i,j);
        end
    end
end

%% Superimposing Images
newyu=I_modulated;
out=im + newyu*0.1;
% figure;
% imshow(out);
% title('Bar Code Content');
%
%% Boundary Padded Picode Image: Padding matrix with zeros. Increases rows & cols
zer_v=zeros(1,116,3);

v=vertcat(zer_v,out);
v=vertcat(zer_v,v);

v=vertcat(v,zer_v);
v=vertcat(v,zer_v);

zer_h=zeros(120,1,3);

h=horzcat(zer_h,v);
h=horzcat(zer_h,h);

h=horzcat(h,zer_h);
h=horzcat(h,zer_h);

% figure;
% imshow(h);
% title('Zero Padded Boundary Image');

%% Finder Pattern Embedding Stage
m=120;
n=120;

for k=1:3
    for i=1:m
        for j=3:3:n-2
            if i==1||i==2
                h(i,j,k)=255;
            end
        end
    end
end

% figure;
% imshow(h,'InitialMag','fit');
% title('Timer Pattern Generated for horizontal row Image');

for k=1:3
    for i=3:3:m-2
        for j=1:n
            if j==n||j==n-1
                h(i,j,k)=255;
            end
        end
    end
end

h(1:2,118,3)=255;
h(118,119:120,3)=255;

figure;
imshow(h);
title('PiCode Image');

imwrite(h,'picoimg.jpg');

%% secrete image embedding

im_d=h; % generated pi code
figure;
imshow(im_d);
title('Picode Input Image/ cover image ');

% secrete image
m=imread('cameraman.tif');
% m=imread('coins.png');

% m=imread('mono.jpg');

img = im_d ;
% seperating components of Pi code image R, G,B
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel

%

m = imresize(m,[120 120]);

red = imresize(red,[120 120]);

green = imresize(green,[120 120]);

blue = imresize(blue,[120 120]);
% subplot(3,3,1)
figure ,imshow(m);
title('SECRETE IMAGE');


size(m);
% m1=im2bw(m);
imwrite(m,'scrt.jpg');

s=[] ;
%      a=1;
%      b=1;
kk1=red;
kk2=green;
kk3=blue;
k1k=[];
k2k=[];
k3k=[];
sv1=[];
tic

% lsb encoding part
[a, b, c]=size(kk1);
for i=1:a
    for j=1:b
        sv1=de2bi(m(i,j),8);

        % red chan
        s1=de2bi(kk1(i,j),8);
        %         s1(1:3)=sv1(1:3);
        b1=bitxor(s1(1),s1(4));
        b1=bitxor(b1,s1(6));
        b1=bitxor(b1,s1(7));

        if b1~=sv1(1)
            s1(1)=~s1(1);
        end

        b2=bitxor(s1(1),s1(4));
        b2=bitxor(b2,s1(5));
        b2=bitxor(b2,s1(7));

        if b2~=sv1(2)
            s1(2)=~s1(2);
        end

        b3=bitxor(s1(3),s1(5));
        b3=bitxor(b3,s1(6));
        b3=bitxor(b3,s1(7));

        if b3~=sv1(3)
            s1(3)=~s1(3);
        end

        if (b1~=sv1(1))&&(b3~=sv1(3))
           s1(6)=~s1(6);
        end

        if (b1~=sv1(1))&&(b2~=sv1(2))
           s1(4)=~s1(4);
        end

        if (b3~=sv1(3))&&(b2~=sv1(2))
               s1(5)=~s1(5);
        end

        % if (b3~=sv1(3))&&(b1~=sv1(1))
        %        s1(6)=~s1(6);
        % end

        if (b1~=sv1(1))&&(b3~=sv1(3))&&(b2~=sv1(2))
               s1(7)=~s1(7);
        end

        s2=de2bi(kk2(i,j),8);

        %         s2(1:3)=sv1(4:6);
        b11=bitxor(s2(1),s2(4));
        b11=bitxor(b11,s2(6));
        b11=bitxor(b11,s2(7));

        if b11~=sv1(4)
            s2(1)=~s2(1);
        end

        b22=bitxor(s2(1),s2(4));
        b22=bitxor(b22,s2(5));
        b22=bitxor(b22,s2(7));

        if b22~=sv1(5)
            s2(2)=~s2(2);
        end

        b33=bitxor(s2(3),s1(5));
        b33=bitxor(b33,s1(6));
        b33=bitxor(b33,s1(7));

        if b33~=sv1(6)
            s2(3)=~s2(3);
        end

        if (b11~=sv1(4))&&(b33~=sv1(6))
            s2(6)=~s2(6);
        end

        if (b11~=sv1(4))&&(b22~=sv1(5))
            s2(4)=~s2(4);
        end

        if (b33~=sv1(6))&&(b22~=sv1(5))
            s2(5)=~s2(5);
        end

        if (b33~=sv1(6))&&(b11~=sv1(4))
            s2(6)=~s2(6);
        end

        if (b11~=sv1(4))&&(b33~=sv1(6))&&(b22~=sv1(5))
            s2(7)=~s2(7);
        end

        s3=de2bi(kk3(i,j),8);

        bb1=bitxor(s3(1),s3(2));
        bb2=bitxor(s3(2),s3(3));

        if (bb1~=sv1(7))&& (bb2==sv1(8))
            s3(1)=~s3(1);
        end

        if (bb2~=sv1(8))&&(bb1==sv1(7))
            s3(3)=~s3(3);
        end

        if (bb1~=sv1(7))&& (bb2~=sv1(8))
            s3(2)=~s3(2);

        end
        %         s3(1:2)=sv1(7:8);


        % s(1,1:7)=sv

        s1 = int8(s1);
        s2 = int8(s2);
        s3 = int8(s3);

        d1 = bi2de(s1);
        kk1(i,j)=d1;

        d2 = bi2de(s2);
        kk2(i,j)=d2;

        d3 = bi2de(s3);
        kk3(i,j)=d3;
    end
end

figure
back_to_original_img = cat(3, kk1, kk2, kk3);
imshow(back_to_original_img)
title('Secrete image embedded')

figure
imshow(back_to_original_img)
imwrite(back_to_original_img,'embdim.jpg');


% decoding part % =========================================================================

kk1 = back_to_original_img(:,:,1); % Red channel
% kk1 = imresize(kk1,[120 120]);

kk2 = back_to_original_img(:,:,2); % Green channel
% kk2 = imresize(kk2,[120 120]);

kk3 = back_to_original_img(:,:,3); % Blue channel
% kk3 = imresize(kk3,[120 120]);

[a, b]=size(kk1);


% % % %
% % vk=de2bi(kk,16)
%  % % BW = im2bw(r);
sv=zeros(1,8);
z=[]
for i=1:a
    for j=1:b
        s1=de2bi(kk1(i,j),8);

        b1=bitxor(s1(1),s1(4));
        b1=bitxor(b1,s1(6));
        b1=bitxor(b1,s1(7));

        b2=bitxor(s1(1),s1(4));
        b2=bitxor(b2,s1(5));
        b2=bitxor(b2,s1(7));

        b3=bitxor(s1(3),s1(5));
        b3=bitxor(b3,s1(6));
        b3=bitxor(b3,s1(7));

        sv(1)=b1;
        sv(2)=b2;
        sv(3)=b3;


        %         sv(1:3)=s1(1:3);
        s2=de2bi(kk2(i,j),8);

        b11=bitxor(s2(1),s2(4));
        b11=bitxor(b11,s2(6));
        b11=bitxor(b11,s2(7));

        b22=bitxor(s2(1),s2(4));
        b22=bitxor(b22,s2(5));
        b22=bitxor(b22,s2(7));

        b33=bitxor(s2(3),s1(5));
        b33=bitxor(b33,s1(6));
        b33=bitxor(b33,s1(7));

        sv(4)=b11;
        sv(5)=b22;
        sv(6)=b33;

        %         sv(4:6)=s2(1:3);
        s3=de2bi(kk3(i,j),8);

        bb1=bitxor(s3(1),s3(2));
        bb2=bitxor(s3(2),s3(3));
        %         sv(7:8)=s3(1:2);
        sv(7)=bb1;
        sv(8)=bb2;
        z(i,j)=bi2de(sv) ;
    end
end

figure
imshow(uint8(z))
title(' Secrete image Decoded')

y=uint8(z);
% y=im2bw(y);
imwrite(y,'rtdimg.jpg');
toc
%  m=im2bw(m)
%
%      pk=uint8(z);
%      pk=im2bw(pk)
% % pkk=histeq(pk)
% %
% figure
% imshow(m);
% figure
% imshow(pk);

% % title('Restored Image');
