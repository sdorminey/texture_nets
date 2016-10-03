require 'image'

local cmd = torch.CmdLine()

cmd:option('-back', '', 'path to static image of background.')
cmd:option('-input', '', 'path to input folder.')
cmd:option('-output', '', 'path to output folder.')
cmd:option('-threshold', 1, 'threshold for pixels')

-- back: image.
-- source, dest: filenames.
local function mask(back, threshold, source, dest)
  local fore = image.load(source, 3):float()

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
  -- Load background.
  local back = image.load(params.back, 3):float()

  -- Sort files in the input path.
  local files = {}
  for file in paths.iterfiles(params.input) do table.insert(files, file) end
  table.sort(files)

  -- Iterate through sorted files, apply the mask.
  for _,file in pairs(files) do
      local source = paths.concat(params.input, file)
      local dest = paths.concat(params.output, file)

      mask(back, params.threshold, source, dest)
  end
end

local params = cmd:parse(arg)
run(params)
