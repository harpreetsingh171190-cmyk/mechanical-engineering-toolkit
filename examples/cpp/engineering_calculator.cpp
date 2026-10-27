// Minimal C++ engineering utility example (C++17)
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

double shaft_torque(double power_w, double rpm) {
    if (!std::isfinite(power_w) || power_w < 0 || !std::isfinite(rpm) || rpm <= 0)
        throw std::invalid_argument("Power must be non-negative and RPM positive.");
    const double omega = 2.0 * std::acos(-1.0) * rpm / 60.0;
    return power_w / omega;
}
int main() {
    try {
        std::cout << "Power (W) and speed (RPM): "; double p, rpm;
        if (!(std::cin >> p >> rpm)) throw std::invalid_argument("Enter numeric inputs.");
        std::cout << std::fixed << std::setprecision(3) << "Torque (N.m): " << shaft_torque(p, rpm) << '\n';
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
