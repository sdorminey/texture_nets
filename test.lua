require 'torch'
require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-batch', false, 'Run in batch mode')
cmd:option('-cycle', false, 'Run in cycle mode')
cmd:option('-compare', false, 'Run in compare mode')
cmd:option('-scale', 1, 'Scale factor for input images.')
cmd:option('-input', '', 'Paths of image to stylize.')
cmd:option('-output', '', 'Path to save stylized image.')
cmd:option('-masks', '', 'Path to masks to use for fader (batch only!)')
cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-fader1', -1, 'Value of fader, in single mode')
cmd:option('-fader2', 0, 'Value of fader, in single mode')
cmd:option('-period', 150, 'Number of frames between fader periods.')
cmd:option('-cpu', false, 'use this flag to run on CPU')
cmd:option('-correct_color', false, 'original colors')
cmd:option('-gain', 1, 'fader gain.')
cmd:option('-start_from', 0, '')
cmd:option('-quad', false, 'Processing quads')

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

-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_yuv = image.rgb2yuv(content)
  generated_y = generated_y - generated_y:mean() + content_yuv[1]:mean()
  local content_uv = content_yuv[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end

-- Takes a 1x4xMxN tensor (1-3 = RGB, 4 = Fader.)
function eval_inner(source)
  local input = source

  -- Mask is needed so we don't get hit by normalization.
  local mask = torch.ByteTensor({{{0, 1}, {1, 0}}})
  mask = mask:repeatTensor(1, input:size(3)/2, input:size(4)/2)
  print(input[1]:size())
  input[1]:select(1, 4):maskedFill(mask, 0)

  -- Stylize
  local stylized = model:forward(input:type(tp)):double()
  stylized = deprocess(stylized[1])

  -- Maybe color correct.
  if params.correct_color then
    stylized = original_colors(input[{1, {1, 3}}]:double(), stylized:double())
  end

  local conv_nodes = model:findModules('cudnn.SpatialConvolution')
  local out1 = conv_nodes[1].output
  local view = out1:squeeze():view(32,-1)
  print('min: ', view:min(), 'max:', view:max())

  return stylized
end

-- Applies a scalar fader.
function eval(source_img, fader)
  -- Insert fader channel.
  local img = source_img:clone()
  img:resize(1, 1+img:size(1), img:size(2), img:size(3))
  img:zero()
  img:sub(1, -1, 1, 3):copy(source_img)
  img[{{}, 4}]:fill(fader)

  return eval_inner(img)
end

-- Applies a mask of faders.
-- Mask is simple MxN tensor.
function eval_mask(source_img, mask)
  local img = source_img:clone()
  img:resize(1, 1+img:size(1), img:size(2), img:size(3))
  img:zero()
  img:sub(1, -1, 1, 3):copy(source_img)
  img[1][4]:copy(mask)

  return eval_inner(img)
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
  print("Processing image " .. source .. " with fader " .. tostring(fader))
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

function apply_with_mask(source, dest, mask_path)
  print("Processing image " .. source .. " with masks from ".. mask_path)
  weights, gradWeights = model:parameters()

  -- Load content image.
  local source_img = image.load(source, 3):float()
  if params.scale ~=1 then
    source_img = image.scale(source_img, source_img:size(3)*params.scale, source_img:size(2)*params.scale)
  end

  -- Load fader mask.
  local fader_mask = image.load(mask_path, 3):float()
  print(fader_mask[1]:size())

  -- Apply style.
  local stylized = eval_mask(source_img, fader_mask[1])

  -- Save
  image.save(dest, torch.clamp(torch.abs(stylized),0,1))
end

if params.cycle then
  local files = {}
  for file in paths.iterfiles(params.input) do table.insert(files, file) end
  table.sort(files)

  local frame_index = 0

  for _,file in pairs(files) do
    local source = paths.concat(params.input, file)
    local dest = paths.concat(params.output, file)

    local index = 0
    if params.quad then
      index = torch.floor(frame_index/4)
    else
      index = frame_index
    end

    -- Fader cycles between fader1 and fader2 every period (fader1 < fader2, duh.)
    local wave = torch.sin((2*math.pi/params.period)*index)
    local wave01 = (wave+1)/2

    local fader = wave01 * params.fader1 + params.fader2 * (1 - wave01)

    apply(source, dest, fader)

    frame_index = frame_index+1
  end
elseif params.batch then
  local files = {}
  for file in paths.iterfiles(params.input) do table.insert(files, file) end
  table.sort(files)

  local counter = params.start_from
  for _,file in pairs(files) do
    if counter > 0 then
      counter = counter-1
    else
      local source = paths.concat(params.input, file)
      local dest = paths.concat(params.output, file)
      local mask = paths.concat(params.masks, file)

      apply_with_mask(source, dest, mask)
    end

    index = index+1
  end
elseif params.compare then
  compare_images(params.input, params.output, params.fader1, params.fader2)
else
  apply(params.input, params.output, params.fader1)
end
