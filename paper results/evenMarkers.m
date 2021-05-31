function [xmarkers,ymarkers, h] = evenMarkers(x,y,NumMarkers, h)
  xmarkers = logspace(log10(x(1)),log10(x(end)),NumMarkers);
  ymarkers = interp1(x,y,xmarkers);
  h = [h, plot(xmarkers, ymarkers, 'd', 'color', 'k', 'MarkerSize', 8, 'linewidth', 5)]; 
end