from treelib import Tree

DROOT = 0

class Tree_Node(object):
    def __init__(self,dep,pos,sen_pos,word_pos,token_id,token_labels = []):
        self.dep = dep
        self.pos = pos
        self.sen_pos = sen_pos
        self.word_pos = word_pos
        self.token_id = token_id
        self.token_labels = token_labels

class Doc_Tree(object):
    def __init__(self):
        self.dp_tree = Tree()
        self.dp_tree.create_node('DROOT', DROOT, data=Tree_Node('','',-1,-1,-1))

    def node_sort(self,node):
        return node.identifier

    def get_all_node(self,cur_node,cnode_list):
        if not cur_node:
            return cnode_list
        child_nodes = self.dp_tree.children(cur_node.identifier)
        for child_node in child_nodes:
            self.get_all_node(child_node,cnode_list)
            cnode_list.append(child_node)

    def build_dp_tree_ltp4(self,words,deps,postags,pnode_id,sen_pos,word_pos_list,token_ids,token_labels):
        for i,dep in enumerate(deps):
            if deps[i][1] == pnode_id:
                self.dp_tree.create_node(words[i],deps[i][0],parent=pnode_id,
                                      data=Tree_Node(deps[i][2],postags[i],sen_pos,word_pos_list[i],token_ids[i],token_labels[i]))
                self.build_dp_tree_ltp4(words,deps,postags,deps[i][0],sen_pos,word_pos_list,token_ids,token_labels)

    def remove_stop_word_nodes_tree(self,stopwords):
        nodes = self.dp_tree.all_nodes()
        for node in nodes:
            if node.tag in stopwords:
                self.dp_tree.link_past_node(node.identifier)
