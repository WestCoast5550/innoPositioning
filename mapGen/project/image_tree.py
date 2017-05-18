from walls_representation import Point


class ImageTree:

    def __init__(self, AP, walls):
        self.Tx = AP
        self.image_tree = self.build_image_tree(walls)

    @staticmethod
    def build_image_tree_layer(Tx, walls):
        layer_points = []
        for wall in walls:
            if wall != Tx.assigned_wall:
                pe = wall.plane_equation  # plane equation
                image_point = Point(0, 0, 0)
                try:
                    common_part = (pe[0] * Tx.x + pe[1] * Tx.y + pe[2] * Tx.z +
                                   pe[3]) / (pe[0] * pe[0] + pe[1] * pe[1] + pe[2] * pe[2])
                except ZeroDivisionError:
                    print("Ouch")
                image_point.x = Tx.x - 2 * pe[0] * common_part
                image_point.y = Tx.y - 2 * pe[1] * common_part
                image_point.z = Tx.z - 2 * pe[2] * common_part
                image_point.assigned_wall = wall
                layer_points.append(image_point)
        return layer_points

    def build_image_tree(self, walls):
        image_tree = Tree(self.Tx)
        tree_position = 0
        #  one layer => one reflection

        new_points = self.build_image_tree_layer(image_tree.tree[tree_position].data, walls)
        image_tree.add_children(new_points, tree_position)
        print("image tree has been built")
        return image_tree


class TreeNode(object):
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children_indices = []


class Tree(object):
    def __init__(self, Tx):
        self.tree = [TreeNode(Tx, None)]

    def add_children(self, children, parent_index):
        for child in children:
            self.tree.append(TreeNode(child, self.tree[parent_index]))
            self.tree[parent_index].children_indices.append(len(self.tree) - 1)

    def get_children(self, parent_index):
        children = []
        children_indices = self.tree[parent_index].children_indices
        for i in children_indices:
            children.append(self.tree[i])
        return children

    def get_children_indices(self, parent_index):
        return self.tree[parent_index].children_indices

