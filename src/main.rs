use std::f32;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Range;

use glam::{vec3, Vec3};
use indicatif::ParallelProgressIterator;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

type RngGen = XorShiftRng;

#[inline(always)]
fn white() -> Vec3 {
    vec3(1.0, 1.0, 1.0)
}

#[inline(always)]
fn black() -> Vec3 {
    vec3(0.0, 0.0, 0.0)
}

#[inline(always)]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        return min;
    }

    if x > max {
        return max;
    }

    return x;
}

pub struct Image {
    data: Vec<Vec3>,
    width: usize,
    height: usize,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Image {
            data: vec![Vec3::default(); width * height],
            width,
            height,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn get(&self, (x, y): (usize, usize)) -> Vec3 {
        let index = y * self.width + x;
        self.data[index]
    }

    pub fn set(&mut self, (x, y): (usize, usize), pixel: Vec3) {
        let index = y * self.width + x;
        self.data[index] = pixel;
    }

    pub fn data_mut(&mut self) -> &mut [Vec3] {
        &mut self.data[..]
    }

    // write as PPM
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "P3")?; // means that colors are in ASCII
        writeln!(w, "{} {}", self.width, self.height)?;
        writeln!(w, "255")?; // max color value

        for line in self.data.chunks(self.width) {
            for pixel in line {
                let r = 256.0 * clamp(pixel.x(), 0.0, 0.999);
                let g = 256.0 * clamp(pixel.y(), 0.0, 0.999);
                let b = 256.0 * clamp(pixel.z(), 0.0, 0.999);
                write!(w, "{} {} {} ", r as u8, g as u8, b as u8)?;
            }
            writeln!(w, "")?;
        }

        Ok(())
    }
}

// Ray which starts at point |origin| and moving to direction |direction|
#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    #[inline(always)]
    fn origin(&self) -> Vec3 {
        self.origin
    }

    #[inline(always)]
    fn direction(&self) -> Vec3 {
        self.direction
    }

    // Calculate position of ray at "time" t
    #[inline(always)]
    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

// determines the "background" color
fn background_color(ray: Ray) -> Vec3 {
    let unit = ray.direction().normalize();
    let t = 0.5 * (unit.y() + 1.0); // scale [-1; 1] -> [0; 1]

    // blend white + blue from bottom to top
    white() * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
}

struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

// Information about ray hit
#[derive(Debug, Clone)]
struct HitRecord {
    t: f32,             // "time" of hit
    p: Vec3,            // point at which hit have happened
    normal: Vec3,       // normal to the surface at point P
    front_face: bool,   // true if ray is outside the object
    material: Material, // material of hit target
}

trait Hit {
    fn hit(&self, ray: Ray, range: Range<f32>) -> Option<HitRecord>;
}

impl Hit for Sphere {
    // let C - center of sphere of radius R
    // then point P is inside shpere if
    // (p - c) * (p - c)<= R*R  # where * is a dot product
    // in our case, point P could be any on the Ray, so
    // Ray hits the sphere if the equation:
    // (o + d*t - c) * (o + d*t - c) - R*R = 0 is solvable # d = direction, o = origin
    //    =>
    // d*t*d*t + 2*d*t*(o - c) + (o - c) * (o - c) - R*R = 0
    #[inline(always)]
    fn hit(&self, ray: Ray, range: Range<f32>) -> Option<HitRecord> {
        let oc = ray.origin() - self.center; // o - c

        // ax^2 + 2bx + c = 0
        let a = ray.direction().length_squared();
        let half_b = oc.dot(ray.direction());
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant.is_sign_negative() {
            return None;
        }

        let root = discriminant.sqrt();
        let t = (-half_b - root) / a;

        let hit = |t| {
            let p = ray.at(t);
            let outward_normal = (p - self.center) / self.radius;
            let front_face = ray.direction().dot(outward_normal).is_sign_negative();

            HitRecord {
                t,
                p,
                normal: if front_face {
                    outward_normal
                } else {
                    -outward_normal
                },
                front_face,
                material: self.material.clone(),
            }
        };

        if range.contains(&t) {
            return Some(hit(t));
        }

        let t = (-half_b + root) / a;
        if range.contains(&t) {
            return Some(hit(t));
        }

        None
    }
}

impl<T> Hit for Vec<T>
where
    T: Hit,
{
    #[inline(always)]
    fn hit(&self, ray: Ray, range: Range<f32>) -> Option<HitRecord> {
        let mut record = None;
        let mut max_t = range.end;

        for object in self {
            if let Some(r) = object.hit(ray, range.start..max_t) {
                max_t = r.t;
                record = Some(r.clone());
            }
        }

        record
    }
}

// trait Material {
//     fn scatter(&self, ray: Ray, hit: HitRecord) -> Option<(Vec3, Ray)>; // (attenuation, scattered)
// }

#[derive(Debug, Clone)]
struct Material {
    kind: MaterialKind,
    albedo: Vec3, // how much this material absorbs from different channels (r, g, b)
    // or simply put - color
    fuzz: f32, // ref_idx for Dielectric
}

#[derive(Debug, Clone, Copy)]
enum MaterialKind {
    Lambertian,
    Metal,
    Dielectric,
}

impl Material {
    #[inline(always)]
    fn scatter(
        &self,
        rng: &mut RngGen,
        ray: Ray,
        at: Vec3,
        normal: Vec3,
        front_face: bool,
    ) -> Option<(Vec3, Ray)> {
        match self.kind {
            MaterialKind::Lambertian => {
                let direction = normal + random_unit_vec3(rng);
                let scattered = Ray {
                    origin: at,
                    direction,
                };
                Some((self.albedo, scattered))
            }
            MaterialKind::Metal => {
                let reflected = reflect(ray.direction.normalize(), normal);
                let scattered = Ray {
                    origin: at,
                    direction: reflected + self.fuzz * random_in_unit_sphere(rng),
                };

                if scattered.direction().dot(normal) > 0.0 {
                    Some((self.albedo, scattered))
                } else {
                    None
                }
            }
            MaterialKind::Dielectric => {
                let attenuation = vec3(1.0, 1.0, 1.0);
                let ref_idx = self.fuzz;
                let etai_over_etat = if front_face { 1.0 / ref_idx } else { ref_idx };

                fn min(a: f32, b: f32) -> f32 {
                    if a > b {
                        b
                    } else {
                        a
                    }
                }

                let unit_direction = ray.direction().normalize();
                let cos_theta = min((-unit_direction).dot(normal), 1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                if etai_over_etat * sin_theta > 1.0 {
                    // reflect
                    let reflected = reflect(unit_direction, normal);
                    let scattered = Ray {
                        origin: at,
                        direction: reflected,
                    };
                    return Some((attenuation, scattered));
                }

                let reflect_prob = schlick(cos_theta, etai_over_etat);
                if rng.gen::<f32>() < reflect_prob {
                    // reflect
                    let reflected = reflect(unit_direction, normal);
                    let scattered = Ray {
                        origin: at,
                        direction: reflected,
                    };
                    return Some((attenuation, scattered));
                }

                let refracted = refract(unit_direction, normal, etai_over_etat);
                let scattered = Ray {
                    origin: at,
                    direction: refracted,
                };
                Some((attenuation, scattered))
            }
        }
    }
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lens_radius: f32,
    u: Vec3,
    v: Vec3,
    w: Vec3,
}

impl Camera {
    // NOTE: fov is vertical, in degrees
    fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        view_up: Vec3,
        fov: f32,
        aspect: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Camera {
        let theta = fov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;

        let w = (lookfrom - lookat).normalize();
        let u = view_up.cross(w).normalize();
        let v = w.cross(u);

        Camera {
            origin: lookfrom,
            lower_left_corner: lookfrom
                - half_width * focus_dist * u
                - half_height * focus_dist * v
                - focus_dist * w,
            horizontal: 2.0 * half_width * focus_dist * u,
            vertical: 2.0 * half_height * focus_dist * v,
            lens_radius: aperture / 2.0,
            u,
            v,
            w,
        }
    }

    #[inline(always)]
    fn ray(&self, u: f32, v: f32, rng: &mut RngGen) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(rng);
        let offset = self.u * rd.x() + self.v * rd.y();

        Ray {
            origin: self.origin + offset,
            direction: self.lower_left_corner + u * self.horizontal + v * self.vertical
                - self.origin - offset,
        }
    }
}

#[inline(always)]
fn reflect(v: Vec3, normal: Vec3) -> Vec3 {
    v - 2.0 * v.dot(normal) * normal
}

#[inline(always)]
fn refract(v: Vec3, normal: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-v).dot(normal);
    let r_out_parallel = etai_over_etat * (v + cos_theta * normal);
    let r_out_perp = -(1.0 - r_out_parallel.length_squared()).sqrt() * normal;
    r_out_parallel + r_out_perp
}

#[inline(always)]
fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    // pow(1.0 - cosine, 5)
    let base = 1.0 - cosine;
    let base_squared = base * base;
    let rhs = base_squared * base_squared * base;
    r0 + (1.0 - r0) * rhs
}

// cos(x) distribution
#[inline(always)]
fn random_unit_vec3(rng: &mut RngGen) -> Vec3 {
    let a = rng.gen::<f32>() * 2.0 * f32::consts::PI;
    let z = -1.0 + rng.gen::<f32>() * 2.0;
    let r = (1.0 - z).sqrt();

    vec3(r * f32::cos(a), r * f32::sin(a), z)
}

#[inline(always)]
fn random_vec3(rng: &mut RngGen, min: f32, max: f32) -> Vec3 {
    Vec3::splat(min) + vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()) * (max - min)
}

#[inline(always)]
fn random_in_unit_sphere(rng: &mut RngGen) -> Vec3 {
    loop {
        let v = random_vec3(rng, -1.0, 1.0);
        if v.length_squared() < 1.0 {
            break v;
        }
    }
}

#[inline(always)]
fn random_in_unit_disk(rng: &mut RngGen) -> Vec3 {
    loop {
        let v = vec3(
            // [-1.0, 1.0]
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
            0.0,
        );

        if v.length_squared() < 1.0 {
            break v;
        }
    }
}

fn random_scene() -> Vec<Sphere> {
    let seed = rand::thread_rng().gen::<u64>();
    let mut rng = RngGen::seed_from_u64(seed);

    let mut spheres = Vec::new();
    spheres.push(Sphere {
        center: vec3(0.0, -1000.0, 0.0),
        radius: 1000.0,
        material: Material {
            kind: MaterialKind::Lambertian,
            albedo: vec3(0.5, 0.5, 0.5),
            fuzz: 0.0,
        }
    });

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen::<f32>();
            let center = vec3(a as f32 + 0.9 * rng.gen::<f32>(), 0.2, b as f32 + 0.9 * rng.gen::<f32>());

            if (center - vec3(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = random_vec3(&mut rng, 0.0, 1.0) * random_vec3(&mut rng, 0.0, 1.0);
                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material {
                            kind: MaterialKind::Lambertian,
                            albedo,
                            fuzz: 0.0,
                        }
                    })
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = random_vec3(&mut rng, 0.5, 1.0);
                    let fuzz = rng.gen::<f32>() / 2.0;
                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material {
                            kind: MaterialKind::Metal,
                            albedo,
                            fuzz
                        }
                    });
                } else {
                    // glass
                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material {
                            kind: MaterialKind::Dielectric,
                            albedo: vec3(0.0, 0.0, 0.0),
                            fuzz: 1.5,
                        }
                    })
                }
            }
        }
    }

    spheres.push(Sphere {
        center: vec3(0.0, 1.0, 0.0),
        radius: 1.0,
        material: Material {
            kind: MaterialKind::Dielectric,
            albedo: vec3(0.0, 0.0, 0.0),
            fuzz: 1.5,
        }
    });

    spheres.push(Sphere {
        center: vec3(-4.0, 1.0, 0.0),
        radius: 1.0,
        material: Material {
            kind: MaterialKind::Lambertian,
            albedo: vec3(0.4, 0.2, 0.1),
            fuzz: 0.0,
        }
    });

    spheres.push(Sphere {
        center: vec3(4.0, 1.0, 0.0),
        radius: 1.0,
        material: Material {
            kind: MaterialKind::Metal,
            albedo: vec3(0.7, 0.6, 0.5),
            fuzz: 0.0,
        }
    });

    spheres
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = Image::new(1600, 800);

    let samples_per_pixel = 500;

    let lookfrom = vec3(13.0, 2.0, 3.0);
    let lookat = vec3(0.0, 0.0, 0.0);
    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let camera = Camera::new(
        lookfrom,
        lookat,
        vec3(0.0, 1.0, 0.0),
        20.0,
        aspect_ratio,
        0.1,
        10.0
    );
    let world = random_scene();

    let height = image.height();
    let width = image.width();

    image
        .data_mut()
        .par_chunks_mut(width)
        .enumerate()
        .progress_count(height as u64)
        .for_each(|(k, line)| {
            let seed = rand::thread_rng().gen::<u64>();
            let mut rng = RngGen::seed_from_u64(seed);

            let j = height - k - 1;
            for i in 0..width {
                let mut pixel = black();
                for _ in 0..samples_per_pixel {
                    let u = (i as f32 + rng.gen::<f32>()) / width as f32;
                    let v = (j as f32 + rng.gen::<f32>()) / height as f32;
                    let mut ray = camera.ray(u, v, &mut rng);

                    const MAX_BOUNCES: usize = 50;

                    let mut i = 0;
                    let mut attenuation = vec3(1.0, 1.0, 1.0);
                    let color = loop {
                        if i > MAX_BOUNCES {
                            break black();
                        }

                        match world.hit(ray, 0.001..f32::INFINITY) {
                            Some(hit) => {
                                match hit.material.scatter(
                                    &mut rng,
                                    ray,
                                    hit.p,
                                    hit.normal,
                                    hit.front_face,
                                ) {
                                    Some((attenuation_increment, scattered)) => {
                                        attenuation *= attenuation_increment;
                                        ray = scattered;
                                        i += 1;
                                    }
                                    None => break black(),
                                }
                            }
                            None => break background_color(ray),
                        }
                    };

                    pixel += color * attenuation;
                }

                let color = pixel / samples_per_pixel as f32;
                let color = vec3(color.x().sqrt(), color.y().sqrt(), color.z().sqrt()); // gamma correction

                line[i] = color;
            }
        });

    let out = File::create("out.ppm")?;
    let mut out = BufWriter::new(out);
    image.write(&mut out)?;
    Ok(())
}
