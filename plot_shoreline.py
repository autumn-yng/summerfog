''' 
Author: Autumn Nguyen
Version: July 2023
'''

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

def plot_from_us_shoreline(bounds):
	shoreline = gpd.read_file("/storage/ngoc54n/us_shoreline/us_medium_shoreline.shp")
	shoreline["geometry"] = shoreline.clip_by_rect(*bounds)
	shoreline = shoreline[~ shoreline["geometry"].is_empty]
	return shoreline

def plot_from_globe_shoreline(bounds):
	# In GSHHS_h_L1, h means high resolution, L1 means land and ocean boundaries.
	shoreline = gpd.read_file("/storage/ngoc54n/global_shoreline/GSHHS_shp/h/GSHHS_h_L1.shp")
	shoreline["geometry"] = shoreline.clip_by_rect(*bounds)
	# Keep only the non-empty items
	shoreline = shoreline[~ shoreline["geometry"].is_empty]
	shoreline.reset_index(inplace=True) # reset the index after removing empty items
	
	# We only want the vector points of the boundaries between land and ocean, not the vector points for all points that are land and all that are oceans
	# so we have to get the Linestrings/MultiLinestrings objects from Polygon/MultiPolygon objects
	# The first two items are MultiPolygons, and the rest are Polygons
	
	for i in range(2):
		shoreline.loc[i, 'geometry'] = MultiPolygon(shoreline.loc[i, 'geometry']).boundary
	for i in range(2, shoreline['geometry'].size):
		shoreline.loc[i, 'geometry'] = Polygon(shoreline.loc[i, 'geometry']).boundary
		# .loc[row_indexer, col_indexer] works, while .iloc[col_indexer][row_indexer] will throw an error of "A value is trying to be set on a copy of a slice from a DataFrame."

	return shoreline

print("Successfully run plot_shoreline.py")