require 'image'
require 'src/content_loss'
require 'src/texture_loss'
local t = require 'datasets/transforms'

require 'loadcaffe'

local ArtisticCriterion, parent = torch.class('nn.ArtisticCriterion', 'nn.Module')

function ArtisticCriterion:__init(params)
   parent.__init(self)

   self.descriptor_net, self.content_modules, self.texture_modules = create_descriptor_net(params, cnn)
   self.gradInput = nil
end

function ArtisticCriterion:updateStrength(texture, content)
  if #self.texture_modules > 0 then
    for k, module in pairs(self.texture_modules) do
      print('Updating texture ', k, ' to ', texture)
      module.strength = texture
    end
  end

  if #self.content_modules > 0 then
    for k, module in pairs(self.content_modules) do
      print('Updating content ', k, ' to ', content)
      module.strength = content
    end
  end
end

--counter = 0
function ArtisticCriterion:updateStyle(full_texture_image, image_size)
  -- Random crop.
  local texture_image = t.RandomCrop(image_size)(full_texture_image)
--  counter = counter+1
--  image.save('texture-'..tostring(counter)..'.jpg', texture_image)
  texture_image = texture_image:type(dtype):add_dummy()

  -- Compute Gram matrix in each style layer.
  local gram = GramMatrix():type(dtype)

  -- Disable the texture loss modules, so that they just record the output of the texture layer in their output field.
  for k, module in pairs(self.texture_modules) do
    module.active = false
  end

  -- Run network forward to capture texture layer responses.
  self.descriptor_net:forward(texture_image)

  for k, module in pairs(self.texture_modules) do
    print('Updating texture style ', k)

    module.active = true

    -- Assign target to the layer's output when it was in that mode.
    local target = gram:forward(nn.View(-1):type(dtype):setNumInputDims(2):forward(module.output)) -- used to be target_features[1]
    if module.target then
      module.target:resizeAs(target)
      module.target:copy(target)
    else
      module.target = target:clone()
    end

    module.target:div(module.output:nElement())
    module.target = module.target:add_dummy()
  end
end

function ArtisticCriterion:updateOutput(input)

  local pred = input[1]
  local target = input[2]

  -- Compute target content features

  if #self.content_modules > 0 then
    for k, module in pairs(self.texture_modules) do
      module.active = false
    end
    for k, module in pairs(self.content_modules) do
      module.active = false
    end
    self.descriptor_net:forward(target)
  end
  
  -- Now forward with images from generator
  for k, module in pairs(self.texture_modules) do
    module.active = true
  end
  for k, module in pairs(self.content_modules) do
    module.active = true
    module.target:resizeAs(module.output)
    module.target:copy(module.output)
  end
  self.descriptor_net:forward(pred)
  
  local loss = 0
  for _, mod in ipairs(self.content_modules) do
    loss = loss + mod.loss
  end
  for _, mod in ipairs(self.texture_modules) do
    loss = loss + mod.loss
  end

  return loss
end

function ArtisticCriterion:updateGradInput(input, gradOutput)
  self.gradInput= self.gradInput or {nil, input[2].new()}
  self.gradInput[1] = self.descriptor_net:updateGradInput(input[1])
  return self.gradInput
end

function nop()
  -- nop.  not needed by our net
end


function create_descriptor_net(params)

  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):type(dtype)

  local content_layers = params.content_layers:split(",") 
  local texture_layers  = params.texture_layers:split(",")

  -- Set up the network, inserting texture and content loss modules
  local content_modules, texture_modules = {}, {}
  local next_content_idx, next_texture_idx = 1, 1
  local net = nn.Sequential()

  for i = 1, #cnn do
     if next_content_idx <= #content_layers or next_texture_idx <= #texture_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

      if params.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution') then
          print (name, ': padding set to 0')

          layer.padW = 0 
          layer.padH = 0 
      end
      net:add(layer)
   
      ---------------------------------
      -- Content
      ---------------------------------
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)

        local norm = false
        local loss_module = nn.ContentLoss(norm):type(dtype)
        net:add(loss_module)
        table.insert(content_modules, loss_module)
        next_content_idx = next_content_idx + 1
      end
      ---------------------------------
      -- Texture
      ---------------------------------
      if name == texture_layers[next_texture_idx] then
        print("Setting up texture layer  ", i, ":", layer.name)

        local norm = params.normalize_gradients
        local loss_module = nn.TextureLoss(norm):type(dtype)
        
        net:add(loss_module)
        table.insert(texture_modules, loss_module)
        next_texture_idx = next_texture_idx + 1
      end
    end
  end

  net:add(nn.DummyGradOutput())

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
      
  return net, content_modules, texture_modules
end
