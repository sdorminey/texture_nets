require 'torch'
require 'image'

local cmd = torch.CmdLine()

cmd:option('-back', nil, 'path to static image of background.')
cmd:option('-input', nil, 'path to input folder.')
cmd:option('-output', nil, 'path to output folder.')

cmd:option('-fore', 'examples/foreground.jpg', 'full scene image')
cmd:option('-back', 'examples/background.jpg', 'scene to subtract')
--cmd:option('-shift', 'examples/shift.jpg', 'output image')
cmd:option('-additive', 'examples/additive.jpg', 'just the scene subtract')
--cmd:option('-mask', 'examples/mask.jpg', 'output image')
cmd:option('-scene_threshold', 40, 'how many shades does foreground differ by')
cmd:option('-component_threshold', .4, 'differentiating connected components')
cmd:option('-framecount', 5, 'number of frames to iterate through')

local function mask(source, dest)
  local fore = image.load(source, 3, double)
  local back = image.load(params.back, 3, double)

  local delta = torch.add(fore, -1, back)
  image.save('examples/delta.jpg', delta)  
  delta = torch.abs(delta)
  image.save(params.additive, delta)

  -- normalized greyscale
  local grey_delta = image.rgb2y(delta)
  image.save(params.additive, grey_delta)
end

local function run(params)
  -- Sort files in the input path.
  local files = {}
  for file in paths.iterfiles(params.input) do table.insert(files, file) end
  table.sort(files)

  -- Iterate through sorted files, apply the mask.
  for _,file in pairs(files) do
      local source = paths.concat(params.input, file)
      local dest = paths.concat(params.output, file)

      mask(source, dest)
  end
end

local function main(params)

  for img_index=1,framecount
  local fore = image.load(params.fore, 3, double)
  local back = image.load(params.back, 3, double)
  local scene_threshold = params.scene_threshold * 1.0/255  
  local empty = fore:clone()
  
  -- absolute value of background subtracted from foreground
  -- is how different pixels are, minus the scene_threshold
  -- a shift in one direction registers same as shift in other
  -- because we don't care about your checkerboard shirt
  
  local delta = torch.add(fore, -1, back)
  image.save('examples/delta.jpg', delta)  
  delta = torch.abs(delta)
  image.save(params.additive, delta)
  
  local s = delta:clone()
  s = s:fill(scene_threshold)
  delta = torch.add(s, -1, delta)

  -- normalized greyscale
  local grey_delta = image.rgb2y(delta)
  image.save(params.additive, grey_delta)

--  local mask = torch.sign(delta)
--  image.save('examples/mask.jpg', mask)
--  local components = grey_delta:clone()
--  imgraph.connectcomponents(components, grey_delta, 0.5, true)
--  local mask = torch.sign(grey_delta)
--  image.save('examples/mask.jpg', mask)
--  linear * mask + linear * inverse of mask = the thing i guess?
  
end

local params = cmd:parse(arg)
main(params)
