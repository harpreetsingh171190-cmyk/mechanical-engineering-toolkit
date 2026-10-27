% Mass-spring-damper simulation: m*x'' + c*x' + k*x = 0
% Run in MATLAB. Educational model; SI units.
m = 10; k = 1000; c = 5;
x0 = 0.05; v0 = 0; tspan = [0 10];
[t, y] = ode45(@(t,y) [y(2); -(c*y(2)+k*y(1))/m], tspan, [x0; v0]);
plot(t, y(:,1), 'LineWidth', 1.5); grid on;
xlabel('Time (s)'); ylabel('Displacement (m)'); title('Mass-spring-damper response');
