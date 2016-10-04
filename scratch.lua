require 'image'

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
  fade = torch.FloatTensor(b):zero()
  k = b-1
  fade:apply(function(x) v = ((b/(b-1))*k)/b; k=k-1; return v; end)

  s = fade:repeatTensor(b,1)
  square = torch.add(torch.tril(s), torch.triu(s:t(), 1))

  south = fade:repeatTensor(w, 1):t():clone()
  east = fade:repeatTensor(h, 1):clone()

  box = torch.ones(h+b, w+b)
  box[{{1, h}, {w+1, w+b}}] = east
  box[{{h+1, h+b}, {1, w}}] = south
  box[{{h+1, h+b}, {w+1, w+b}}] = square
  print(box:min(), box:max())
  return box:repeatTensor(3,1,1)
end

-- 0 1
-- 2 3
function apply(q0, q1, q2, q3, border, filename)
  local h = (q0:size(2)-border)*2
  local w = (q0:size(3)-border)*2

  local box = shaded_box(q0:size(1)-border, q0:size(2)-border, border)

  q0 = q0:cmul(box)
  q1 = image.hflip(image.hflip(q1):cmul(box))
  q2 = image.vflip(image.vflip(q2):cmul(box))
  q3 = image.hflip(image.vflip(image.vflip(image.hflip(q3)):cmul(box)))

  local out = torch.Tensor(q0:size(1), h, w)
  out[{{}, {1, h/2}, {1, w/2}}] = q0
  out[{{}, {1, h/2}, {w/2+1, w}}] = q1
  out[{{}, {h/2+1, h}, {1, w/2}}] = q2
  out[{{}, {h/2+1, h}, {h/2+1, h}}] = q3

  return out
end

torch.save('square.t7', make_square(16))
local box = shaded_box(796/2, 1216/2, 128/2)
image.save('scratch.png', box)
