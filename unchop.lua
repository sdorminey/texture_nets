-- Un-chops an image into four quadrants, with a border.

require 'image'

local cmd = torch.CmdLine()

cmd:option('-input', '', 'Path to input images.')
cmd:option('-quads', '', 'Path to transformed quadrants made from the input images.')
cmd:option('-output', '', 'Path to output images.')
cmd:option('-start_from', 0, '')

local params = cmd:parse(arg)

function make_square(b)
  local square = torch.FloatTensor(b,b)
  local k = 0
  square:apply(
    function(x)
      local p = torch.floor(k / square:stride(1))
      local q = k % square:stride(1)
      k=k+1
      return (b - torch.sqrt(p*p + q*q))/b
    end)
  return square
end

function shaded_box(h, w, b)
  local fade = torch.FloatTensor(b):zero()
  local k = b-1
  fade:apply(function(x) v = ((b/(b-1))*k)/b; k=k-1; return v; end)

  local s = fade:repeatTensor(b,1)
  local square = torch.add(torch.tril(s:t()), torch.triu(s, 1))
  square[{{square:size(1)/2+1, -1}, {square:size(2)/2+1, -1}}]:zero() -- Reduces the overlap.

  local south = fade:repeatTensor(w, 1):t():clone()
  local east = fade:repeatTensor(h, 1):clone()

  local box = torch.ones(h+b, w+b):float()
  box[{{1, h}, {w+1, w+b}}] = east
  box[{{h+1, h+b}, {1, w}}] = south
  box[{{h+1, h+b}, {w+1, w+b}}] = square
  print(box:min(), box:max())
  box = box:repeatTensor(3,1,1)
  return box
end

-- 0 1
-- 2 3
function apply(q0, q1, q2, q3, border)
  local h = (q0:size(2)-border)*2
  local w = (q0:size(3)-border)*2

  local box = shaded_box(q0:size(2)-border*2, q0:size(3)-border*2, border*2)

  local q0 = torch.cmul(q0, box)
  local q1 = image.hflip(torch.cmul(image.hflip(q1), box))
  local q2 = image.vflip(torch.cmul(image.vflip(q2), box))
  local q3 = image.hflip(image.vflip(torch.cmul(image.vflip(image.hflip(q3)), box)))

  local out = torch.Tensor(q0:size(1), h, w):float()
  out[{{}, {1, h/2+border}, {1, w/2+border}}]:add(q0)
  out[{{}, {1, h/2+border}, {w/2+1-border, w}}]:add(q1)
  out[{{}, {h/2+1-border, h}, {1, w/2+border}}]:add(q2)
  out[{{}, {h/2+1-border, h}, {w/2+1-border, w}}]:add(q3)
  print(out:min(), out:max())

  return out
end

function unchop(source_filename)
  local source = image.load(paths.concat(params.input, source_filename), 3):float()

  local q0 = image.load(paths.concat(params.quads, source_filename .. '.0.jpg'), 3):float()
  local q1 = image.load(paths.concat(params.quads, source_filename .. '.1.jpg'), 3):float()
  local q2 = image.load(paths.concat(params.quads, source_filename .. '.2.jpg'), 3):float()
  local q3 = image.load(paths.concat(params.quads, source_filename .. '.3.jpg'), 3):float()

  print(q0:size(2), source:size(2)/2)

  local target = apply(q0, q1, q2, q3, q0:size(2) - (source:size(2)/2))
  image.save(paths.concat(params.output, source_filename), target)
end

-- Sort files in the input path.
local files = {}
for file in paths.iterfiles(params.input) do table.insert(files, file) end
table.sort(files)

local counter = params.start_from
-- Iterate through sorted files, apply the mask.
for _,file in pairs(files) do
  if counter > 0 then
    counter = counter-1
  else
    print(file)
    unchop(file)
  end
end
