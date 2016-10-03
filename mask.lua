require 'image'

local cmd = torch.CmdLine()

cmd:option('-backs', '', 'path to static images of background.')
cmd:option('-input', '', 'path to input folder.')
cmd:option('-output', '', 'path to output folder.')
cmd:option('-threshold', 1, 'threshold for pixels')

-- back: image.
-- source, dest: filenames.
local function mask(back, threshold, source, dest)
  local fore = image.load(source, 3):float()

  if back == nil then
    back = torch.FloatTensor():resizeAs(fore)
    back:zero()
  end

  local delta = torch.add(fore, -1, back)
  delta = torch.abs(delta)

  -- normalized greyscale
  local grey_delta = image.rgb2y(delta)

  -- threshold
  local function run_threshold(x)
    if x < threshold then
      return 0
    else
      return x
    end
  end
  grey_delta:apply(run_threshold)

  image.save(dest, grey_delta)
end

local function run(params)
  -- Load background. Start out with totally black.
  local back = nil

  -- Sort backs. File names match when to switch.
  local back_index = 1
  local backs = {}
  for file in paths.iterfiles(params.backs) do table.insert(backs, file) end
  table.sort(backs)

  -- Sort files in the input path.
  local files = {}
  for file in paths.iterfiles(params.input) do table.insert(files, file) end
  table.sort(files)

  -- Iterate through sorted files, apply the mask.
  for _,file in pairs(files) do
    if file == backs[back_index] then
      back_index = back_index+1
      back = image.load(paths.concat(params.backs, back[back_index], 3)):float()
    end

    local source = paths.concat(params.input, file)
    local dest = paths.concat(params.output, file)

    mask(back, params.threshold, source, dest)
  end
end

local params = cmd:parse(arg)
run(params)
