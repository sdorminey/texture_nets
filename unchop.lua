-- Un-chops an image into four quadrants, with a border.

require 'image'

local cmd = torch.CmdLine()

cmd:option('-input', '', 'Path to input images.')
cmd:option('-quads', '', 'Path to transformed quadrants made from the input images.')
cmd:option('-output', '', 'Path to output images.')

local params = cmd:parse(arg)

function ourmul(a, b)
  local temp = a:clone():zero()
  temp:addcmul(2, a, b)
  return temp
end

function shaded_box(h, w, b)
  grad = torch.FloatTensor(h+b):zero()
  k = h+b+1
  grad:apply(function(x) k=k-1; return k/(h+b); end)

  down = grad:repeatTensor(w+b, 1):t():clone()
  right = grad:repeatTensor(w+b, 1):clone()

  print(w-h)
  print(down:size())
  down:triu(w-h)[{{1, h}, {1, w}}]:zero()
  right:t():tril(w-h-1)[{{1, h}, {1, w}}]:zero()

  down:add(right)
  down[{{1, h}, {1, w}}]:fill(1)
  return down:repeatTensor(3,1,1)
end

function unchop(source_filename)
  local source = image.load(paths.concat(params.input, source_filename), 3):float()

  local q0 = image.load(paths.concat(params.quads, source_filename .. '.0.jpg'), 3):float()
  local q1 = image.load(paths.concat(params.quads, source_filename .. '.1.jpg'), 3):float()
  local q2 = image.load(paths.concat(params.quads, source_filename .. '.2.jpg'), 3):float()
  local q3 = image.load(paths.concat(params.quads, source_filename .. '.3.jpg'), 3):float()

  local target = source:zero()

  local h = source:size(2)
  local w = source:size(3)
  local b = q0:size(3) - source:size(3)/2

  -- Add in whole images, including overlapping borders.
  print(shaded_box(h/2, w/2,b):size())
  local b0 = ourmul(q0, shaded_box(h/2, w/2, b))
  local b1 = image.hflip(q1)
  b1 = ourmul(b1, shaded_box(h/2, w/2, b))
  b1 = image.hflip(b1)
  local b2 = image.vflip(q2)
  b2 = ourmul(b2, shaded_box(h/2, w/2, b))
  b2 = image.vflip(q2)
  local b3 = image.hflip(q3)
  b3 = image.vflip(b3)
  b3 = ourmul(b3,shaded_box(h/2, w/2, b))
  b3 = image.vflip(b3)
  b3 = image.hflip(b3)

  target[{{}, {1, h/2+b}, {1, w/2+b}}]:add(b0)
--  target[{{}, {1, h/2+b}, {w/2-b+1, w}}]:add(b1)
--  target[{{}, {h/2-b+1, h}, {1, w/2+b}}]:add(b2)
--  target[{{}, {h/2-b+1, h}, {w/2-b+1, w}}]:add(b3)

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
