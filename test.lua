require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-batch', false, 'Run in batch mode')
cmd:option('-input', '', 'Paths of image to stylize.')
cmd:option('-output', '', 'Path to save stylized image.')
cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-fader', 1, 'Value of fader, in single mode')
cmd:option('-period', 50, 'Number of frames between fader periods.')
cmd:option('-cpu', false, 'use this flag to run on CPU')

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

function style_image(source, dest, fader)
  print("Processing image " .. source .. "with fader " .. tostring(fader))
  weights, gradWeights = model:parameters()
  print(weights[1])

  -- Load
  local source_img = image.load(source, 3):float()

  -- Insert fader channel.
  local img = source_img:clone()
  img:resize(1, 1+img:size(1), img:size(2), img:size(3))
  img:zero()
  img:sub(1, -1, 1, 3):copy(source_img)
  img:select(2,4):fill(fader)

  -- Stylize
  local input = img
  local stylized = model:forward(input:type(tp)):double()
  stylized = deprocess(stylized[1])

  -- Save
  image.save(dest, torch.clamp(stylized,0,1))
end

if params.batch then
    local files = {}
    for file in paths.iterfiles(params.input) do table.insert(files, file) end
    table.sort(files)

    for _,file in pairs(files) do
      local source = paths.concat(params.input, file)
      local dest = paths.concat(params.output, file)

      -- Fader cycles between 0 and 2 every period.
      local fader = torch.sin((2*math.pi/params.period)*index)+1

      style_image(source, dest, fader)

      index = index+1
    end
else
  style_image(params.input, params.output, params.fader)
end
