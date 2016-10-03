require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-batch', false, 'Run in batch mode')
cmd:option('-compare', false, 'Run in compare mode')
cmd:option('-scale', 1, 'Scale factor for input images.')
cmd:option('-input', '', 'Paths of image to stylize.')
cmd:option('-output', '', 'Path to save stylized image.')
cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-fader1', -1, 'Value of fader, in single mode')
cmd:option('-fader2', -0.5, 'Value of fader, in single mode')
cmd:option('-period', 50, 'Number of frames between fader periods.')
cmd:option('-cpu', false, 'use this flag to run on CPU')
cmd:option('-original_colors', false, 'original colors')

local params = cmd:parse(arg)

-- Load model and set type
local model = torch.load(params.model_t7)

if params.cpu then 
  tp = 'torch.FloatTensor'
else
  require 'cutorch'
  require 'cunn'
  require 'cudnn'

  tp = 'torch.CudaTensor'
  model = cudnn.convert(model, cudnn)
end

model:type(tp)
model:evaluate()

local index = 0

function eval(source_img, fader)
  -- Insert fader channel.
  local img = source_img:clone()
  img:resize(1, 1+img:size(1), img:size(2), img:size(3))
  img:zero()
  img:sub(1, -1, 1, 3):copy(source_img)

  -- Mask is needed so we don't get hit by normalization.
  local mask = torch.ByteTensor({{{0, 1}, {1, 0}}})
  mask = mask:repeatTensor(1, img:size(3)/2, img:size(4)/2)
  img[1]:select(1, 4):maskedFill(mask, fader)

  -- Stylize
  local input = img
  local stylized = model:forward(input:type(tp)):double()
  stylized = deprocess(stylized[1])

  local conv_nodes = model:findModules('cudnn.SpatialConvolution')
  local out1 = conv_nodes[1].output
  local view = out1:squeeze():view(32,-1)
  print('min: ', view:min(), 'max:', view:max())

  return stylized
end

function compare_images(source, dest, fader1, fader2)
  print("Processing image " .. source .. "with faders " .. tostring(fader1) .. ' and ' .. tostring(fader2))
  weights, gradWeights = model:parameters()

  -- Load
  local source_img = image.load(source, 3):float()

  local v1 = eval(source_img, fader1)
  local v2 = eval(source_img, fader2)
  local diff = v1-v2
  print('diff min:', diff:min(), 'diff max:', diff:max())

  -- Save
  image.save(dest, torch.clamp(torch.abs(diff),0,1))
end

function apply(source, dest, fader)
  print("Processing image " .. source .. "with fader " .. tostring(fader))
  weights, gradWeights = model:parameters()

  -- Load
  local source_img = image.load(source, 3):float()
  if params.scale ~=1 then
    source_img = image.scale(source_img, source_img:size(3)*params.scale, source_img:size(2)*params.scale)
  end

  local stylized = eval(source_img, fader)

  -- Save
  image.save(dest, torch.clamp(torch.abs(stylized),0,1))
end

if params.batch then
    local files = {}
    for file in paths.iterfiles(params.input) do table.insert(files, file) end
    table.sort(files)

    for _,file in pairs(files) do
      local source = paths.concat(params.input, file)
      local dest = paths.concat(params.output, file)

      -- Fader cycles between fader1 and fader2 every period (fader1 < fader2, duh.)
      local wave = torch.sin((2*math.pi/params.period)*index)
      local fader = params.fader1 + (params.fader2-params.fader1) * (wave+1)/2

      apply(source, dest, fader)

      index = index+1
    end
elseif params.compare then
  compare_images(params.input, params.output, params.fader1, params.fader2)
else
  apply(params.input, params.output, params.fader1)
end
