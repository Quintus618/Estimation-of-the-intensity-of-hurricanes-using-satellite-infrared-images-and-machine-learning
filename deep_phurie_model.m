clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nc_file = 'data.nc' ;

% Get information about the file
ncinfo(nc_file);

% Display the file
ncdisp(nc_file);

% Open and read a NC file with particular variable
hurricane_data = ncread(nc_file,'peaks');

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
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%