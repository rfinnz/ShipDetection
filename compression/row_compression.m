clear all;
clc;
%  

%%%%%%%%%%%%%%% Image Input %%%%%%%%%%%
%p = imread('test.png');
p = imread('frame3.jpg');%% Reading image file
p = rgb2gray(p); %%Gray-scale conversion
p = im2double(p);  %%unit8 to double conversion
% imshow(p);
 
 
%%%%%%%%%%%% Extracting row from the image %%%%%%%%%

%%% t= 256*0+1:512*1; %For extracting 512 samples from the signal. 
t= 256*0+1:576*1; %% length of row
data(1,:)= p(1,t); %%Copying data

%%% Selection of compression ratio
compression_ratio = 50;
 
%%%%%%%%%%%%%%%%%%%% Declaration %%%%%%%%%%%%%%%%%%%%%%%%%%%%
n2 = 64;  %number of samples in each segment
c_r = (100 - compression_ratio)/100; 
m2 = round(n2*c_r);  %number of samples in compressed segment
% CR = (1-m2/n2)*100;

%%%%%%%%%%%%%% Initial Calculations %%%%%%%%%%%%%%%%%%%%%%%%
 
%l = length(data(1,:));
l = length(p(:,1));
Kronecker_Factor = length(p)/n2; %row length is considered as a reference Kronecker Factor
 
n1 = n2*Kronecker_Factor; 
m1 = m2*Kronecker_Factor; 

%%% Storage space allocation %%%

test_y1 = zeros(l,m1);

%%%%%%%%%%%%%%%%%%%%%% Formation of Measurement Matrix %%%%%%%%%%%%%%
 
%Main_Measurement_Matrix=((1/m2)^2)*randn(round(m2),n2); %Normal Distribution 
Main_Measurement_Matrix = dbbd(n2,m2);

%%For finding matrix of required size 
Kron_Measurement_Matrix=kron(eye(n1/n2),Main_Measurement_Matrix);  %%numpy.kron(A,B) is python equivalent
% A1=Kron_Measurement_Matrix*dict1;
% A2=Main_Measurement_Matrix*dict2;

for row = 1:l
% for edit = 1:500 %Do not activate  
 
%signal = p(column,:);
 
%Main_Measurement_Matrix=((1/m2)^2)*binornd(n2,0.5,round(m2),n2); %%%Bernoulli Distribution, as both normal and binomial validates RIP! [R3]
%Main_Measurement_Matrix=((1/m2)^2)*randn(m2,n2); %%%Gaussian distribution  
 
%%%%%%%%%%%%%%%%%%%%% Compression %%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
signal= p(row,:);
 
for i=1:l/n1
      
       y1=Kron_Measurement_Matrix*signal((i-1)*n1+1:n1*i,1);
       test_y1(row,:) = y1;
       
end

 
%  end
end

t_y1 = uint8(255 * mat2gray(test_y1));  %%%unit8 conversion

figure, imshow(t_y1)

%%%%%%%%%%%%%%%%%%%%%% File Storage %%%%%%%%%%%%%%%%%%%%%%%

% filename = 'frame3';
% figure, imshow(t_y1)
% imwrite(t_y1,['c_',filename,'_',num2str(compression_ratio),'.jpg']);

