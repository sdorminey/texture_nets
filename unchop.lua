-- Un-chops an image into four quadrants, with a border.

require 'image'

local cmd = torch.CmdLine()

cmd:option('-input', '', 'Path to input images.')
cmd:option('-quads', '', 'Path to transformed quadrants made from the input images.')
cmd:option('-output', '', 'Path to output images.')

local params = cmd:parse(arg)

function unchop(source_filename)
  local source = image.load(paths.concat(params.input, source_filename), 3):float()

  local q0 = image.load(paths.concat(params.quads, source_filename .. '.0.jpg'), 3):float()
  local q1 = image.load(paths.concat(params.quads, source_filename .. '.1.jpg'), 3):float()
  local q2 = image.load(paths.concat(params.quads, source_filename .. '.2.jpg'), 3):float()
  local q3 = image.load(paths.concat(params.quads, source_filename .. '.3.jpg'), 3):float()

  local target = source:zero()

  local h = source:size(2)
  local w = source:size(3)
  local border = q0:size(3) - source:size(3)/2

  -- Add in whole images, including overlapping borders.
  target[{{}, {1, h/2+border}, {1, w/2+border}}]:add(q0)
  target[{{}, {1, h/2+border}, {w/2-border+1, w}}]:add(q1)
  target[{{}, {h/2-border+1, h}, {1, w/2+border}}]:add(q2)
  target[{{}, {h/2-border+1, h}, {w/2-border+1, w}}]:add(q3)

  target[{{}, {h/2-border+1, h/2+border}, {}}]:div(2)
  target[{{}, {}, {w/2-border+1, w/2+border}}]:div(2)
--  target[{{}, {h/2-border+1, h/2+border}, {w/2-border+1, w/2+border}}]:div(2) -- square in the middle.

  image.save(paths.concat(params.output, source_filename), target)
end

-- Sort files in the input path.
local files = {}
for file in paths.iterfiles(params.input) do table.insert(files, file) end
table.sort(files)

-- Iterate through sorted files, apply the mask.
for _,file in pairs(files) do
  print(file)
  unchop(file)
end
