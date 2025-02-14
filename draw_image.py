from PIL import Image
import numpy as np
import random
import copy

class quantized_color_generator(object):
    def __init__(self, 
                 inner_dim=10, 
                 outer_padding=5, 
                 kernel_size=30, 
                 pixel_size=2,
                 lightness_mode="CIE",
                 debug=False,
                 balance_mode="Lightness",
                 restrict_ligntness=True
                ):
        
        # 2*30*20*5=6000
        # canvas size
        self.inner_dim = inner_dim
        self.outer_padding = outer_padding

        # default pattern size
        self.kernel_size = kernel_size
        self.pixel_size = pixel_size

        # default color and lightness
        self.base_colors = [
            [255, 0, 0], # red
            [0, 255, 0], # green
            [0, 0, 255], # blue
            [255, 255, 255], # white
            [0, 0, 0], # black
        ]
        self.lightness_mode = lightness_mode
        self.red_lightness = self.rgb_to_lightness(255, 0, 0)
        self.green_lightness = self.rgb_to_lightness(0, 255, 0)
        self.blue_lightness = self.rgb_to_lightness(0, 0, 255)
        print(self.red_lightness, self.green_lightness, self.blue_lightness)

        self.debug=debug
        self.balance_mode = balance_mode
        self.restrict_ligntness = restrict_ligntness

    def hex_to_rgb(self, hex):
        r = int(hex[1:3], 16)
        g = int(hex[3:5], 16)
        b = int(hex[5:7], 16)
        return r, g, b
    
    def rgb_to_linear(self, c):
        c = c / 255.0 # normalize to [0, 1]
        if c <= 0.04045:
            return c / 12.92 
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    def relative_luminance_to_lightness(self, Y, Y_n=1.0):
        if Y / Y_n <= (6 / 29) ** 3:
            return (Y / Y_n) * 903.3
        else:
            return 116 * (Y / Y_n) ** (1/3) - 16
    
    def rgb_to_lightness(self, r, g, b):
        if(self.lightness_mode == "CIE"):
            # map rgb value to anothor color space for color mixture
            r_lin, g_lin, b_lin = self.rgb_to_linear(r), self.rgb_to_linear(g), self.rgb_to_linear(b)
            # Calculate relative luminance
            Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin # HDTV
            # Calculate lightness
            L = self.relative_luminance_to_lightness(Y) 
            return L / 100.0
        elif(self.lightness_mode == "HSL"):
            return 0.5 * (max(r,g,b)+min(r,g,b)) / 255.0
        elif(self.lightness_mode == "HSV"):
            return max(r,g,b) / 255.0

    def rgb_to_hue(self, r, g, b):
        norm_r = (r - min(r, g, b))
        norm_g = (g - min(r, g, b))
        norm_b = (b - min(r, g, b))
        sum_rgb = norm_r + norm_g + norm_b
        if(sum_rgb == 0):
            return 0, 0, 0
        else:
            return norm_r / sum_rgb, norm_g / sum_rgb, norm_b / sum_rgb

    def rgb_to_saturation(self, r, g, b):
        if(max(r,g,b) == 0):
            return 0
        else:
            return  (max(r,g,b) - min(r,g,b)) / (max(r,g,b))

    def rgb_to_hls(self, r, g, b):
        hue = self.rgb_to_hue(r,g,b)
        lightness = self.rgb_to_lightness(r,g,b)
        saturation = self.rgb_to_saturation(r,g,b)
        return hue, lightness, saturation

    def probs_2_hls(self, color_probs):
        hue = self.rgb_to_hue(color_probs[0], color_probs[1], color_probs[2])
        sat = 1 - color_probs[3] - color_probs[4]
        lightness = self.red_lightness * color_probs[0] + self.green_lightness * color_probs[1] + self.blue_lightness * color_probs[2] + color_probs[3]
        return hue, sat, lightness

    def generate_random_hue(self):
        hue = [0, 0, 0]
        while(max(hue) == 0):
            hue = self.rgb_to_hue(random.random() * 255, random.random() * 255, random.random() * 255) 
        return hue

    def lightness_range(self, hue, sat):        
        base_lightness = self.red_lightness * hue[0] + self.green_lightness * hue[1] + self.blue_lightness * hue[2]
        max_lightness = base_lightness * (1 - sat) + sat
        min_lightness = base_lightness * (1 - sat)
        return min_lightness, max_lightness
    
    def calculate_ratio(self, hue, target_lightness, target_saturation):
        # color part lightness
        if(max(hue) == 0 or target_saturation == 0):
            white = target_lightness
            black = 1 - target_lightness
            return [0,0,0,white,black] # saturation is 0, only black and white
            
        norm_r = hue[0]
        norm_g = hue[1]
        norm_b = hue[2]
        color_lightness = (self.red_lightness * norm_r + self.green_lightness * norm_g + self.blue_lightness * norm_b) / (norm_r + norm_g + norm_b)
        
        # lightness adjustment 
        if(target_lightness > color_lightness):
            white = (target_lightness - color_lightness) / (1 - color_lightness)
            black = 0
        else:
            black = 1 - target_lightness / color_lightness
            white = 0
        
        color_ratio = target_saturation    
        current_total = white + black + color_ratio
        
        if(current_total > 1):
            if(self.balance_mode == "Lightness"):
                color_ratio = 1 - white - black
            elif(self.balance_mode == "Balanced"):
                color_ratio /= current_total
                white /= color_ratio
                black /= color_ratio
            else:
                white = max(white - (current_total - 1), 0)
                black = max(black - (current_total - 1), 0)
        else:
            delta = 1 - current_total
            white += delta * color_lightness
            black += delta * (1 - color_lightness)
    
        red = color_ratio * norm_r / ((norm_r + norm_g + norm_b))
        green = color_ratio * norm_g / ((norm_r + norm_g + norm_b))
        blue = color_ratio * norm_b / ((norm_r + norm_g + norm_b))
    
        return [red, green, blue, white, black]

    def sample_color(self, color_probs):
        index = random.choices(range(5), weights=color_probs, k=1)[0]
        chosen_color = self.base_colors[index]
        return chosen_color

    def draw_single_hex(self, hex_color, map_width, map_height, verbose=False):
        r, g, b = self.hex_to_rgb(hex_color)
        h, l, s = self.rgb_to_hls(r, g, b)
        return self.draw_single_hls(h, l, s, map_width, map_height, verbose)
        
    def draw_single_hls(self, h, l, s, map_width, map_height, verbose=False):
        print(f"rendering block with lightness {l:.2f}, hue [{h[0]:.2f},{h[1]:.2f},{h[2]:.2f}], saturation {s:.2f}. ")

        map_width = int(map_width)
        map_height = int(map_height)
        
        color_probs = self.calculate_ratio(h, l, s)
        red_ratio, green_ratio, blue_ratio, white_ratio, black_ratio = color_probs[0], color_probs[1], color_probs[2], color_probs[3], color_probs[4]
        print(f"estimated {red_ratio*100:.2f}% red, {green_ratio*100:.2f}% green, {blue_ratio*100:.2f}% blue, {white_ratio*100:.2f}% white, {black_ratio*100:.2f}% black")
        
        pattern = []
        for i in range(self.kernel_size):
            curr_row = []
            for j in range(self.kernel_size):
                curr_color = self.sample_color(color_probs)
                curr_row.extend([curr_color for x in range(self.pixel_size)])
            pattern.extend([curr_row for x in range(self.pixel_size)])
        pattern = np.array(pattern, dtype=np.uint8)


        row_map = np.concat([pattern for i in range(map_height)], axis = 0)
        full_map = np.concat([row_map for i in range(map_width)], axis = 1)                

        if(verbose):
            origin_image_map = np.array([[[r,g,b] for x in range(map_width * self.kernel_size * self.pixel_size)] for y in range(map_height * self.kernel_size * self.pixel_size)], dtype=np.uint8)
            origin_image_map = Image.fromarray(origin_image_map, mode="RGB")
            # display(origin_image_map)
    
            result = {
                "Target Hue" : self.rgb_to_hue(r,g,b),
                "Actual Hue" : self.rgb_to_hue(color_probs[0], color_probs[1], color_probs[2]),
                "Target Saturation": self.rgb_to_saturation(r,g,b),
                "Actual Saturation": float(np.sum(color_probs[:3])),
                "Target Lightness": self.rgb_to_lightness(r,g,b),
                "Actual Lightness": self.red_lightness * color_probs[0] + self.green_lightness * color_probs[1] + self.blue_lightness * color_probs[2] + color_probs[3]
            }
    
            for key in result:
                print(key, result[key])

        if(self.debug):
            origin_image_map = np.array([[[r,g,b] for x in range(map_width * self.kernel_size * self.pixel_size)] for y in range(map_height * self.kernel_size * self.pixel_size)], dtype=np.uint8)
            return origin_image_map, color_probs

        return full_map, color_probs
    
    def draw_block(self, color1, color2, color3, color4):
        r1, g1, b1 = self.hex_to_rgb(color1)
        h1, l1, s1 = self.rgb_to_hls(r1, g1, b1)
        r2, g2, b2 = self.hex_to_rgb(color2)
        h2, l2, s2 = self.rgb_to_hls(r2, g2, b2)
        r3, g3, b3 = self.hex_to_rgb(color3)
        h3, l3, s3 = self.rgb_to_hls(r3, g3, b3)
        r4, g4, b4 = self.hex_to_rgb(color4)
        h4, l4, s4 = self.rgb_to_hls(r4, g4, b4)

        average_l = (l1 + l2 + l3 + l4) / 4

        inner_up, _ = self.draw_single_hls(
            h1, average_l, s1,
            map_width = self.inner_dim, 
            map_height = self.inner_dim / 2, 
        )
        inner_down, _ = self.draw_single_hls(
            h2, average_l, s2,
            map_width = self.inner_dim, 
            map_height = self.inner_dim / 2, 
        )
        outer_up, color_up = self.draw_single_hls(
            h3, average_l, s3,
            map_width = self.inner_dim + 2 * self.outer_padding, 
            map_height = self.inner_dim / 2 + self.outer_padding, 
        )
        outer_down, color_down = self.draw_single_hls(
            h4, average_l, s4,
            map_width = self.inner_dim + 2 * self.outer_padding, 
            map_height = self.inner_dim / 2 + self.outer_padding, 
        )
        
        final_outer = np.concatenate([outer_up, outer_down], axis=0)
        final_inner = np.concatenate([inner_up, inner_down], axis=0)
    
        begin = self.outer_padding * self.kernel_size * self.pixel_size
        end = (self.outer_padding + self.inner_dim) * self.kernel_size * self.pixel_size
        
        final_outer[begin:end, begin:end] = final_inner
        
        return final_outer, [color_up, color_down]

    def draw_dependent(self, 
                       up_dep1=None, 
                       up_dep2=None, 
                       down_dep1=None, 
                       down_dep2=None
                      ):         

        # upper outer block color
        hue_or_sat = random.random() # hue-1, sat-2 for hue > 0.5; hue-2, sat-1 for hue <= 0.5
        upper_outer_hue = self.generate_random_hue()
        upper_outer_sat = random.random()
        if(up_dep1 is not None):
            if(hue_or_sat > 0.5):
                upper_outer_hue = self.rgb_to_hue(up_dep1[0], up_dep1[1], up_dep1[2])
            else:
                upper_outer_sat = 1 - up_dep1[3] - up_dep1[4]
        if(up_dep2 is not None):
            if(hue_or_sat > 0.5):
                upper_outer_sat = 1 - up_dep2[3] - up_dep2[4]
            else:
                upper_outer_hue = self.rgb_to_hue(up_dep2[0], up_dep2[1], up_dep2[2])
        
        # lower outer block color
        hue_or_sat = random.random() # hue-1, sat-2 for hue > 0.5; hue-2, sat-1 for hue <= 0.5
        lower_outer_hue = self.generate_random_hue()
        lower_outer_sat = random.random()
        if(down_dep1 is not None):
            if(hue_or_sat > 0.5):
                lower_outer_hue = self.rgb_to_hue(down_dep1[0], down_dep1[1], down_dep1[2])
            else:
                lower_outer_sat = 1 - down_dep1[3] - down_dep1[4]
        if(down_dep2 is not None):
            if(hue_or_sat > 0.5):
                lower_outer_sat = 1 - down_dep2[3] - down_dep2[4]
            else:
                lower_outer_hue = self.rgb_to_hue(down_dep2[0], down_dep2[1], down_dep2[2])

        # decide lightness
        if(self.restrict_ligntness):
            min_lightness_up, max_lightness_up = self.lightness_range(upper_outer_hue, upper_outer_sat)
            min_lightness_lo, max_lightness_lo = self.lightness_range(lower_outer_hue, lower_outer_sat)
            lightness = max(min(1, max_lightness_up, max_lightness_lo, np.random.normal(0.5, 0.1)), 0, min_lightness_lo, min_lightness_up)
        else:
            lightness = max(min(1, np.random.normal(0.5, 0.1)), 0)

        # draw all blocks
        outer_up, color_up = self.draw_single_hls(
            h=upper_outer_hue,
            l=lightness,
            s=upper_outer_sat,
            map_width = self.inner_dim + 2 * self.outer_padding, 
            map_height = self.inner_dim / 2 + self.outer_padding, 
        )
        outer_down, color_down = self.draw_single_hls(
            h=lower_outer_hue,
            l=lightness,
            s=lower_outer_sat,
            map_width = self.inner_dim + 2 * self.outer_padding, 
            map_height = self.inner_dim / 2 + self.outer_padding, 
        )
        inner_up, _ = self.draw_single_hls(
            h=self.generate_random_hue(),
            l=lightness,
            s=random.random(),
            map_width = self.inner_dim, 
            map_height = self.inner_dim / 2, 
        )
        # draw lower inner block
        inner_down, _ = self.draw_single_hls(
            h=self.generate_random_hue(),
            l=lightness,
            s=random.random(),
            map_width = self.inner_dim, 
            map_height = self.inner_dim / 2, 
        )
        
        final_outer = np.concatenate([outer_up, outer_down], axis=0)
        final_inner = np.concatenate([inner_up, inner_down], axis=0)
    
        begin = self.outer_padding * self.kernel_size * self.pixel_size
        end = (self.outer_padding + self.inner_dim) * self.kernel_size * self.pixel_size
        
        final_outer[begin:end, begin:end] = final_inner
    
        return final_outer, [color_up, color_down]


    def draw_center_column(
                            self,
                            color1, 
                            color2, 
                            color3, 
                            color4
                        ):
        # record colors
        grid_colors = {}
        
        # center img
        center_img, center_colors = self.draw_block(color1, color2, color3, color4)
        grid_colors["center"] = center_colors
    
        # up
        up_img, up_colors = self.draw_dependent(
            down_dep1=grid_colors["center"][0]
        )
        grid_colors["up"] = up_colors
    
        # down
        down_img, down_colors = self.draw_dependent(
            up_dep1=grid_colors["center"][1]
        )
        grid_colors["down"] = down_colors
    
        # concat
        center_column = np.concatenate([up_img, center_img, down_img])
        return center_column, grid_colors

    def draw_surround_column(self, grid_colors):
        # center
        center_img, center_colors = self.draw_dependent(
            up_dep1=grid_colors["center"][0], 
            down_dep1=grid_colors["center"][1], 
        )
        grid_colors["center"] = center_colors
        
        # up
        up_img, up_colors = self.draw_dependent(
            up_dep1=grid_colors["up"][0], 
            down_dep1=grid_colors["up"][1], 
            down_dep2=grid_colors["center"][0], 
        )
        grid_colors["up"] = up_colors
    
        # down
        down_img, down_colors = self.draw_dependent(
            down_dep1=grid_colors["down"][1],
            up_dep1=grid_colors["down"][0],
            up_dep2=grid_colors["center"][1],
        )
        grid_colors["down"] = down_colors
    
        # concat
        center_column = np.concatenate([up_img, center_img, down_img])
        return center_column, grid_colors

    def draw_all_figures(self, color1, color2, color3, color4): 
        center_column, center_colors = self.draw_center_column(color1, color2, color3, color4)
        left_column, left_colors = self.draw_surround_column(copy.deepcopy(center_colors))
        right_column, right_colors = self.draw_surround_column(copy.deepcopy(center_colors))
        left2_column, left2_colors = self.draw_surround_column(copy.deepcopy(left_colors))
        right2_column, right2_colors = self.draw_surround_column(copy.deepcopy(right_colors))

        if(self.debug):
            for key in center_colors:
                print(f"center-{key}-up", self.probs_2_hls(center_colors[key][0]))
                print(f"center-{key}-lo", self.probs_2_hls(center_colors[key][1]))
            for key in left_colors:
                print(f"left-{key}-up", self.probs_2_hls(left_colors[key][0]))
                print(f"left-{key}-lo", self.probs_2_hls(left_colors[key][1]))
            for key in right_colors:
                print(f"right-{key}-up", self.probs_2_hls(right_colors[key][0]))
                print(f"right-{key}-lo", self.probs_2_hls(right_colors[key][1]))

        # concat
        all_figure = np.concatenate([left2_column, left_column, center_column, right_column, right2_column], axis=1)
        all_figure = Image.fromarray(all_figure)

        # all_figure.save("current_image.png")

        return all_figure