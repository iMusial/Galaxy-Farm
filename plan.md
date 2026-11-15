# PLAN
Simple List:
1. Tell farmers when to plant
2. Use data trends to make predictions
3. Use serpAPI in combination with data to build a crop rotation strategy
4. Recommend a crop rotation strategy

## 1. Establish a List of Usable data
- Determine seasonal trends from data?
- Data should be accessible via APIs so we aren't downloading large files

Good Data Sources So Far:
- Imerg: rainfall reliability and precipitation
- Accessible via [API](insert url here)
- CDL: Crop and crop rotation
- Accessible via [API](https://nassgeodata.gmu.edu/CropScape/)

## 2. Use Data to Make Visualizations
- Streamlite website handles this

## 3. Build a Crop Rotation Strategy
- The farmer doesn't see the strategy until the following steps are complete
	1. The farmer gives the crop name
	2. SerpAPI fills in crop information (?? maybe)
		- The unique needs those crops have, like specific nutrients and water consumption
	3. Imerg data is used to predict when and how much the farmer should water
	4. CDL data and relevant sources are used to predict what specific nutrients will be depleted
	5. Crops can be rotated based on the nutrients data
	6. Generate an (annual/seasonal? monthly?) crop rotation plan plan

## 4. Recommend a Crop Rotation Strategy
- Drop a pin on your farm in the map
- Get an overview of current conditions there
- Ask the farmer what crops they're using
- prism_explorer generates a rotation plan by fetching a range of data
- Show the farmer in a structured format when and what to rotate

- Somewhere in there, we must include serpAPI. Maybe instead of coding in each crop, we can use serpAPI to verify the crop exists, then fetch necessary information on the crops?

Todo: 
- Add more functions to prism_explorer
	- Ask the farmer what crops are being used
	- Somewhere, store specific information relevant to our data. For example, a farmer enters corn, we need to know how much water corn needs and use Imerg data to tell the farmer when to water, and how much.
	- Generate the crop rotation plan to aid the farmers crop rotation decision, while considering seasonal changes as well.
		a. Should the rotation plan be an annual plan? One season or a few months at a time?

- Clean up repository
	- User-friendly repo format
	- READMEs with good instructions
	- Documentation of how and what prism_explorer does
	
