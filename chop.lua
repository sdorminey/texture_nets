-- Chops an image into four quadrants, with a border.

require 'image'

local cmd = torch.CmdLine()

cmd:option('-border', 0, 'Pixel width of segment border.')
cmd:option('-input', '', 'Path to input images.')
cmd:option('-output', '', 'Path to output images.')

local params = cmd:parse(arg)

function chop(source_path, border)
  local source_filename = paths.basename(source_path)

  local source = image.load(source_path, 3):float()
  local h = source:size(2)
  local w = source:size(3)

  local q0 = source[{{}, {1, h/2+border}, {1, w/2+border}}]
  local q1 = source[{{}, {1, h/2+border}, {w/2-border+1, w}}]
  local q2 = source[{{}, {h/2-border+1, h}, {1, w/2+border}}]
  local q3 = source[{{}, {h/2-border+1, h}, {w/2-border+1, w}}]

  image.save(paths.concat(params.output, source_filename .. '.0.jpg'), q0)
  image.save(paths.concat(params.output, source_filename .. '.1.jpg'), q1)
  image.save(paths.concat(params.output, source_filename .. '.2.jpg'), q2)
  image.save(paths.concat(params.output, source_filename .. '.3.jpg'), q3)
end

-- Sort files in the input path.
local files = {}
for file in paths.iterfiles(params.input) do table.insert(files, file) end
table.sort(files)

-- Iterate through sorted files, apply the mask.
for _,file in pairs(files) do
  print(file)
  local source = paths.concat(params.input, file)
  chop(source, params.border)
end
