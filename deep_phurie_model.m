clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hurricane localized in the sea
nc_file_1 = './HURSAT-B1/2004/HURSAT_b1_v06_2004247N10332_IVAN_c20170721/2004247N10332.IVAN.2004.09.09.2100.18.GOE-12.114.hursat-b1.v06.nc';

% Hurricane where its image has zero pixel intensity
nc_file_2 = './HURSAT-B1/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721/2005236N23285.KATRINA.2005.08.26.0600.66.GOE-10.057.hursat-b1.v06.nc';

% Hurricen localized after landfall
nc_file_3 = './HURSAT-B1/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721/2005236N23285.KATRINA.2005.08.29.1500.39.GOE-12.084.hursat-b1.v06.nc';

rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

% Get information about the file
ncinfo(nc_file_1)

% Display the file
ncdisp(nc_file_1);

% Open and read a NC file with particular variable
% satellite IR image of a hurricane where we apply a colormap
hurricane_IR_image_1 = ncread(nc_file_1,'IRWIN');
imshow(hurricane_IR_image_1);
title("IVAN Hurricane IR")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_1 = ncread(nc_file_1,'VSCHN');
imshow(hurricane_visible_image_1);
title("IVAN Hurricane visible")
figure


% Get some precise information about nc file
hurricane_wind_speed_1 = ncread(nc_file_1,'WindSpd');
hurricane_long_cent_1 = ncread(nc_file_1,'archer_lon');
hurricane_lat_cent_1 = ncread(nc_file_1,'archer_lat');
hurricane_sat_name_1 = ncreadatt(nc_file_1,"/","Satellite_Name");

hurricane_IR_image_2 = ncread(nc_file_2,'IRWIN');
imshow(hurricane_IR_image_2);
title("KATRINA Hurricane IR 1")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_2 = ncread(nc_file_2,'VSCHN');
imshow(hurricane_visible_image_2);
title("KATRINA Hurricane visible 1 zero pixel intensity")
figure

hurricane_wind_speed_2 = ncread(nc_file_2,'WindSpd');
hurricane_long_cent_2 = ncread(nc_file_2,'archer_lon');
hurricane_lat_cent_2 = ncread(nc_file_2,'archer_lat');
hurricane_sat_name_2 = ncreadatt(nc_file_2,"/","Satellite_Name");

hurricane_IR_image_3 = ncread(nc_file_3,'IRWIN');
imshow(hurricane_IR_image_3);
title("KATRINA Hurricane IR landfall")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_3 = ncread(nc_file_3,'VSCHN');
imshow(hurricane_visible_image_3);
title("KATRINA Hurricane visible landfall")

hurricane_wind_speed_3 = ncread(nc_file_3,'WindSpd');
hurricane_long_cent_3 = ncread(nc_file_3,'archer_lon');
hurricane_lat_cent_3 = ncread(nc_file_3,'archer_lat');
hurricane_sat_name_3 = ncreadatt(nc_file_3,"/","Satellite_Name");

% Convolutional neural network trained
%h5_file = './deep-Phurie-master/model/model.h5'
%h5disp(h5_file)

% Download HURSAT-B1 dataset from 2004 to 2009

% Filter the dataset (preprocessing)

% Split into training set and test set

% Model



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS AND PROCEDURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Download HURSAT-B1 dataset from the url
% Create HURSAT-B1 folder
% Inside this folder, create a folder for each year from 2004 to 2009
% HURSAT-B1
% => 2004 2005 2006 2007 2008 2009
% Fill each folder with respective hurricanes data
% Extract hurricane data
% WARNING : the size of the dataset is 21Go
function download_HURSAT_B1()
end

% Annotate the center of the hurricane
% Box that indicates the center of the hurricane
function annotated_hurricane_center(image)
end

% Detect and remove landfall image
% return True if landfall is detected
function detected = remove_landfall(image)
end

% Detect and remove zero pixel intensity and negative pixel in an image
% if detected, return True
function detected = pixel_treatment(image)
end

% preprocessing phase
function preprocessed_data = preprocessing()

    % Browse folders and select each file
    
        % remove images with zero pixel intensity and negative pixels
    
        % remove images with landfall
    
        % filter only images taken by GOES 12
    
        % resize image
    
        % add image to preprocessed_data
end

% Convolutional Neural Network
function model = convo_neural_network()
    layers = [
        
    ]
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%